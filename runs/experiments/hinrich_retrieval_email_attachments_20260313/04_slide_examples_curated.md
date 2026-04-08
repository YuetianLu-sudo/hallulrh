## success

- model: `gemma_7b_it`
- relation: `company_hq`
- query_id: `1833`
- slide question: Where is Ipsos MORI headquartered?
- original prompt: Where are the headquarters of Ipsos MORI?
- gold object: London
- strict(train-only): rank 1 / 141
- strict top-5: London; Stockholm; Chicago; Berlin; Athens
- all-candidates: rank 1 / 163
- all-candidates top-5: London; Stockholm; Chicago; Berlin; Tokyo

## failure

- model: `gemma_7b_it`
- relation: `company_hq`
- query_id: `2199`
- slide question: Where is Jacobs Engineering Group headquartered?
- original prompt: Where are the headquarters of Jacobs Engineering Group?
- gold object: Pasadena
- strict(train-only): rank 134 / 141
- strict top-5: London; Chicago; Berlin; Stockholm; Tokyo
- all-candidates: rank 146 / 163
- all-candidates top-5: London; Chicago; Berlin; Tokyo; Stockholm

## strict_miss_but_all_good

- model: `gemma_7b_it`
- relation: `company_ceo`
- query_id: `1539`
- slide question: Who is the CEO of Goldman Sachs?
- original prompt: Who is the CEO of Goldman Sachs?
- gold object: David Solomon
- strict(train-only): gold not in candidate bank
- all-candidates: rank 1 / 287
- all-candidates top-5: David Solomon; Martin Brudermüller; Mark Avery; Kevin Johnson; Jane Fraser

