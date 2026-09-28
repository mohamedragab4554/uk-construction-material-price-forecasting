# Data

All series are monthly price indices (2015 = 100) published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Source attribution: *Office for National Statistics* and *Department for Business and Trade, Building Materials and Components Statistics*, licensed under the OGL v3.0.

| File | Series | Source | Provenance check |
|---|---|---|---|
| `raw/metal_doors_windows_ppi_EW7C_2010_2025.csv` | PPI output domestic, C2512 Doors and windows of metal | ONS, series **EW7C** (MM22) | Values match the ONS page (Feb 2010 87.0 … Jan 2025 177.7) |
| `raw/ceramic_tiles_ppi_2010_2025.csv` | Ceramic tiles and flags PPI | ONS PPI; downloaded in April 2025 for the coursework, **series code not recorded** | Does **not** match the import index ERTB, so it is probably the domestic-output index. Treated as unverified |
| `raw/materials_index_2014_2024.csv` | Construction material price index used for the concrete and brick estimates | ONS PPI; downloaded for the coursework, **series code not recorded** | Not the DBT Table 1a sector indices, and not JUW8. Treated as unverified |
| `raw/dbt_table1a_material_price_indices_2019_2023.csv` | Construction material price indices by sector: new housing, other new work, repair and maintenance, all work | DBT *Monthly Statistics of Building Materials and Components*, Table 1a, Bulletin 587 (7 Feb 2024) | Extracted from the published workbook. Late-2023 months are provisional (`provisional = True`) |

The raw CSVs are kept as downloaded. The coursework files use a `2014 Jan` / `Feb` / `Mar` layout that `matprice.data.load_ons_csv` parses correctly (see the README for why that matters).
