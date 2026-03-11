use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use std::collections::{BTreeMap, BTreeSet};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Token, Type, Visibility};

struct Param {
    name: Ident,
    ty: Ident,
    indices: Vec<char>,
}

struct ReturnSpec {
    ty: Type,
    indices: Vec<char>,
}

struct EinsumFnDef {
    vis: Visibility,
    name: Ident,
    params: Vec<Param>,
    ret: ReturnSpec,
}

fn parse_index_chars(input: ParseStream) -> syn::Result<Vec<char>> {
    input.parse::<Token![/]>()?;
    let ident: Ident = input.parse()?;
    let chars: Vec<char> = ident.to_string().chars().collect();
    if chars.is_empty() {
        return Err(syn::Error::new(ident.span(), "index string must not be empty"));
    }
    Ok(chars)
}

impl Parse for EinsumFnDef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vis: Visibility = input.parse()?;
        let name: Ident = input.parse()?;

        let content;
        syn::parenthesized!(content in input);

        let mut params = Vec::new();
        while !content.is_empty() {
            let pname: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            let ty: Ident = content.parse()?;
            let indices = parse_index_chars(&content)?;
            params.push(Param {
                name: pname,
                ty,
                indices,
            });
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }

        input.parse::<Token![->]>()?;
        let ret_ty: Type = input.parse()?;

        // optional /indices on return type (absent = scalar output)
        let ret_indices = if input.peek(Token![/]) {
            parse_index_chars(input)?
        } else {
            vec![]
        };

        // optional trailing semicolon
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
        }

        Ok(EinsumFnDef {
            vis,
            name,
            params,
            ret: ReturnSpec {
                ty: ret_ty,
                indices: ret_indices,
            },
        })
    }
}

/// `einsum_fn!(matmul(a: T/ab, b: T/bc) -> T/ac)`
///
/// Generates a function performing Einstein summation. Each single character
/// in the index strings is a dimension. Indices present in inputs but absent
/// from the output are contracted (summed over).
///
/// For tensor outputs (`-> T/ac`), the generated code calls:
/// - `T::new(dims: Vec<usize>) -> T` — construct a zero tensor
/// - `.d[axis]`                       — dimension size
/// - `.get(&[usize]) -> Elem`         — read element
/// - `.set(&[usize], Elem)`           — write element
///
/// For scalar outputs (`-> f32`), no `/indices` on the return type means all
/// indices are contracted. The accumulator is typed as the return type and
/// initialized with `Default::default()`.
///
/// Examples:
///   `einsum_fn!(dot(a: T/i, b: T/i) -> f32)`       — dot product
///   `einsum_fn!(trace(m: T/aa) -> f32)`              — trace
#[proc_macro]
pub fn einsum_fn(input: TokenStream) -> TokenStream {
    let def = syn::parse_macro_input!(input as EinsumFnDef);

    match generate(&def) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn generate(def: &EinsumFnDef) -> syn::Result<proc_macro2::TokenStream> {
    let scalar_output = def.ret.indices.is_empty();

    // Map: index_char -> Vec<(param_index, position_in_param)>
    let mut index_locs: BTreeMap<char, Vec<(usize, usize)>> = BTreeMap::new();

    for (pi, param) in def.params.iter().enumerate() {
        for (pos, &ch) in param.indices.iter().enumerate() {
            index_locs.entry(ch).or_default().push((pi, pos));
        }
    }

    // Validate: every output index must appear in at least one input
    for &ch in &def.ret.indices {
        if !index_locs.contains_key(&ch) {
            return Err(syn::Error::new(
                Span::call_site(),
                format!("output index '{ch}' does not appear in any input"),
            ));
        }
    }

    let free_set: BTreeSet<char> = def.ret.indices.iter().copied().collect();
    let contracted: Vec<char> = index_locs
        .keys()
        .filter(|ch| !free_set.contains(ch))
        .copied()
        .collect();

    if scalar_output && contracted.is_empty() {
        return Err(syn::Error::new(
            Span::call_site(),
            "scalar output but no indices to contract",
        ));
    }

    // Canonical source for each index: first occurrence
    let canonical: BTreeMap<char, (usize, usize)> = index_locs
        .iter()
        .map(|(&ch, locs)| (ch, locs[0]))
        .collect();

    // Idents for dimension variables and loop variables
    let dim_id: BTreeMap<char, Ident> = index_locs
        .keys()
        .map(|&ch| (ch, Ident::new(&format!("dim_{ch}"), Span::call_site())))
        .collect();
    let idx_id: BTreeMap<char, Ident> = index_locs
        .keys()
        .map(|&ch| (ch, Ident::new(&format!("idx_{ch}"), Span::call_site())))
        .collect();

    // 1. Dimension bindings
    let dim_bindings: Vec<_> = index_locs
        .keys()
        .map(|&ch| {
            let d = &dim_id[&ch];
            let (pi, pos) = canonical[&ch];
            let p = &def.params[pi].name;
            let pos_lit = proc_macro2::Literal::usize_unsuffixed(pos);
            quote! { let #d = #p.d[#pos_lit]; }
        })
        .collect();

    // 2. Assertions: matching dimensions for shared indices
    let assertions: Vec<_> = index_locs
        .iter()
        .flat_map(|(&ch, locs)| {
            if locs.len() <= 1 {
                return vec![];
            }
            let (pi0, pos0) = locs[0];
            locs[1..]
                .iter()
                .map(move |&(pi, pos)| {
                    let p1 = &def.params[pi0].name;
                    let l1 = proc_macro2::Literal::usize_unsuffixed(pos0);
                    let p2 = &def.params[pi].name;
                    let l2 = proc_macro2::Literal::usize_unsuffixed(pos);
                    let msg = format!("einsum dimension mismatch for index '{ch}'");
                    quote! { assert_eq!(#p1.d[#l1], #p2.d[#l2], #msg); }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // 3. Build get expressions for each input
    let input_gets: Vec<_> = def
        .params
        .iter()
        .map(|param| {
            let pname = &param.name;
            let idx_exprs: Vec<_> = param
                .indices
                .iter()
                .map(|ch| &idx_id[ch])
                .collect();
            quote! { #pname.get(&[#(#idx_exprs),*]) }
        })
        .collect();

    // Product of all input elements
    let product = if input_gets.len() == 1 {
        input_gets[0].clone()
    } else {
        let first = &input_gets[0];
        let rest = &input_gets[1..];
        quote! { #first #(* #rest)* }
    };

    let ret_ty = &def.ret.ty;

    // 4. Build the loop body — differs for scalar vs tensor output
    let (preamble, loops, epilogue) = if scalar_output {
        // Scalar output: accumulate into a typed variable, return it
        let mut inner = quote! { acc += #product; };

        // All indices are contracted (free set is empty)
        let all_indices: Vec<char> = index_locs.keys().copied().collect();
        for &ch in all_indices.iter().rev() {
            let i = &idx_id[&ch];
            let d = &dim_id[&ch];
            inner = quote! {
                for #i in 0..#d {
                    #inner
                }
            };
        }

        let preamble = quote! { let mut acc: #ret_ty = Default::default(); };
        let epilogue = quote! { acc };
        (preamble, inner, epilogue)
    } else {
        // Tensor output: construct result, nested loops with set()
        let output_dims: Vec<_> = def.ret.indices.iter().map(|ch| &dim_id[ch]).collect();
        let output_idx: Vec<_> = def.ret.indices.iter().map(|ch| &idx_id[ch]).collect();

        let body = if contracted.is_empty() {
            // No contraction — direct assignment (e.g. transpose)
            quote! {
                result.set(&[#(#output_idx),*], #product);
            }
        } else {
            // With contraction — accumulate per output element
            let mut inner = quote! { acc += #product; };

            for &ch in contracted.iter().rev() {
                let i = &idx_id[&ch];
                let d = &dim_id[&ch];
                inner = quote! {
                    for #i in 0..#d {
                        #inner
                    }
                };
            }

            quote! {
                let mut acc = Default::default();
                #inner
                result.set(&[#(#output_idx),*], acc);
            }
        };

        // Wrap in free-index loops
        let mut loops = body;
        for &ch in def.ret.indices.iter().rev() {
            let i = &idx_id[&ch];
            let d = &dim_id[&ch];
            loops = quote! {
                for #i in 0..#d {
                    #loops
                }
            };
        }

        let preamble = quote! { let mut result = #ret_ty::new(vec![#(#output_dims),*]); };
        let epilogue = quote! { result };
        (preamble, loops, epilogue)
    };

    // 5. Assemble function
    let fn_name = &def.name;
    let vis = &def.vis;
    let param_decls: Vec<_> = def
        .params
        .iter()
        .map(|p| {
            let name = &p.name;
            let ty = &p.ty;
            quote! { #name: &#ty }
        })
        .collect();

    Ok(quote! {
        #[allow(non_snake_case)]
        #vis fn #fn_name(#(#param_decls),*) -> #ret_ty {
            #(#dim_bindings)*
            #(#assertions)*
            #preamble
            #loops
            #epilogue
        }
    })
}
