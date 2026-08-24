
| Folder | Dataset | Domain | Kenapa dipilih |
|---|---|---|---|
| `conditionalqa/` | ConditionalQA (ACL 2022) | Kebijakan publik UK (gov.uk) | Paling mirip struktur domain asli (dokumen kebijakan pemerintah) |
| `cuad/` | CUAD | Kontrak komersial/legal | ~50% query *not answerable* — bagus buat uji FRR/FAR |
| `financebench/` | FinanceBench | Audit keuangan (SEC filings) | Sesuai Future Work paper (financial audits) |
| `pubmedqa/` | PubMedQA (PQA-L) | Kesehatan/biomedis | Sesuai Future Work paper (health regulations) |
| `pakdwi/` | Data internal pak Dwi | Domain kustom | Data privat, di luar KB utama |

Judge semua pakai **GPT-5.4-mini** (independen dari generator Gemini) —
sama protokol dengan Table 6 di manuskrip utama.

## Cara run (tiap dataset, dari dalam foldernya masing-masing)

KB tiap dataset **sudah pernah dibangun** (folder `kb_*/` sudah ada) — jadi langkah 1 (build KB) **tidak perlu diulang** kecuali mau bikin KB baru dari nol atau ganti jumlah sampel.

### 1. ConditionalQA
```bash
cd AFTER-REVIEW/cross-domain/conditionalqa
python 01_build_conditionalqa_kb.py          # cuma kalau KB belum ada / mau rebuild
python 02_run_conditionalqa_test.py                        # default: standard_rag, selfrag, arda_sr
python 02_run_conditionalqa_test.py --methods arda_sr       # cuma ARDA-SR
python 02_run_conditionalqa_test.py --smoke                 # 6 query dulu buat sanity check
```

### 2. CUAD
```bash
cd AFTER-REVIEW/cross-domain/cuad
python 01_build_cuad_kb.py [--n 150]
python 02_run_cuad_test.py
python 02_run_cuad_test.py --methods arda_sr
python 02_run_cuad_test.py --smoke
```

### 3. FinanceBench
```bash
cd AFTER-REVIEW/cross-domain/financebench
python 01_build_financebench_kb.py [--n 150]
python 02_run_financebench_test.py
python 02_run_financebench_test.py --methods arda_sr
python 02_run_financebench_test.py --smoke
```

### 4. PubMedQA
```bash
cd AFTER-REVIEW/cross-domain/pubmedqa
python 01_build_pubmedqa_kb.py
python 02_run_pubmedqa_test.py
python 02_run_pubmedqa_test.py --methods arda_sr
python 02_run_pubmedqa_test.py --smoke
```

### 5. Pakdwi (data internal)
```bash
cd AFTER-REVIEW/cross-domain/pakdwi
python 01_build_pakdwi_kb.py
python 02_run_pakdwi_test.py
python 02_run_pakdwi_test.py --methods arda_sr
python 02_run_pakdwi_test.py --smoke              # 10 item dulu
python 02_run_pakdwi_test.py --skip-judge          
```