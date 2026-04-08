## success

- model: `gemma_7b_it`
- relation: `company_hq`
- query_id: `1833`
- question: Where are the headquarters of Ipsos MORI?
- gold object: London
- strict(train-only): rank 1 / 141
- strict top-5: London; Stockholm; Chicago; Berlin; Athens
- all-candidates: rank 1 / 163
- all-candidates top-5: London; Stockholm; Chicago; Berlin; Tokyo

## failure

- model: `gemma_7b_it`
- relation: `company_hq`
- query_id: `2199`
- question: Where are the headquarters of Jacobs Engineering Group?
- gold object: Pasadena
- strict(train-only): rank 134 / 141
- strict top-5: London; Chicago; Berlin; Stockholm; Tokyo
- all-candidates: rank 146 / 163
- all-candidates top-5: London; Chicago; Berlin; Tokyo; Stockholm

## strict_miss_but_all_good

- model: `gemma_7b_it`
- relation: `company_hq`
- query_id: `2273`
- question: Where are the headquarters of Real Academia de Bellas Artes de San Fernando?
- gold object: Madrid
- strict(train-only): gold not in candidate bank
- all-candidates: rank 6 / 163
- all-candidates top-5: London; Berlin; Chicago; Stockholm; Tokyo

