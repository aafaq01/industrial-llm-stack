# Corpus sources

9 public domain US government engineering documents, 1,115 PDF pages, about 27 MB.
The PDFs are not committed. `docker compose up` (service `rag-ingest`) downloads them into the `rag-corpus`
volume, checks each sha256 against [sources.json](sources.json), then rebuilds the index.
Checked 2026-10-07.

## Licence basis

- NIST: works of US federal employees are not subject to copyright in the US (17 U.S.C. 105).
  Only NIST publications written by NIST staff, or carrying an explicit public domain or "not subject to copyright"
  statement, were kept. Rejected: NIST SP 330 (2019) (BIPM copyright notice), NIST AMS 100-1 (prepared by a contractor),
  NIST AMS 100-8 and NISTIR 8183r1 (non-federal co-authors, no explicit statement), NIST AMS 300-8 (withdrawn),
  NIST TN 2148 (dropped 2026-10-08: no copyright statement in the PDF, the work was funded by a DOE contract, and federal
  employment of the first author could not be confirmed from a citable source).
- NASA: the NTRS record for the handbook has copyright determination `PUBLIC_USE_PERMITTED`
  (https://ntrs.nasa.gov/api/citations/20170001761).
- Outside the US, foreign copyright may apply to US government works. This corpus is for local evaluation only.

| id | Title | Pages | Licence evidence | URL | sha256 |
|----|-------|-------|------------------|-----|--------|
| nasa-sp-2016-6105r2 | NASA Systems Engineering Handbook (NASA/SP-2016-6105 Rev 2) | 356 | NTRS: PUBLIC_USE_PERMITTED | https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf | 3153ae2e53e29452d5997efafe280a5f05cd21b43a047e988a17e1dd5207a38e |
| nist-sp-800-82r3 | NIST SP 800-82r3 Guide to Operational Technology (OT) Security (2023) | 316 | Text: "not subject to copyright in the United States" | https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf | 608f554514d381853e24e0b33123b7080fbd757826f460032690aade63394616 |
| nist-sp-1500-201 | NIST SP 1500-201 Framework for Cyber-Physical Systems: Volume 1, Overview (2017) | 79 | Text: "published in the public domain" | https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1500-201.pdf | e6dc5b1627698624f64013e81abd873951a2af4e75eb6aac3795aa48c42ade77 |
| nist-sp-811 | NIST SP 811 (2008) Guide for the Use of the International System of Units (SI) | 90 | NIST staff authors (Thompson, Taylor); 17 U.S.C. 105 | https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication811e2008.pdf | 788dd8f0bcb0ec06e40c300690266742532a7e760a93be57764b77ed0ef3482f |
| nist-tn-1297 | NIST TN 1297 (1994) Guidelines for Evaluating and Expressing the Uncertainty of NIST Measurement Results | 25 | NIST staff authors (Taylor, Kuyatt); 17 U.S.C. 105 | https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1297.pdf | f2c8e6026d5589a63d492f192b72cd905f554b477a6049532256170aec477e92 |
| nist-tn-1900 | NIST TN 1900 Simple Guide for Evaluating and Expressing the Uncertainty of NIST Measurement Results (2015) | 105 | NIST staff author (Possolo); 17 U.S.C. 105 | https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.1900.pdf | 624d62ac48a0e03f532462c292e0253c3c2ab87434a6ad41aed76b0687cab45a |
| nist-ams-300-1 | NIST AMS 300-1 Reference Architecture for Smart Manufacturing Part 1: Functional Models (2016) | 47 | NIST staff authors (Barkmeyer, Wallace); 17 U.S.C. 105 | https://nvlpubs.nist.gov/nistpubs/ams/NIST.AMS.300-1.pdf | 926b30dc7ea372096b4b027ddacf794ae5d08b343c0d0346c13271dae5928a8e |
| nist-ams-200-9 | NIST AMS 200-9 The Manufacturing Cost Guide: A Primer (2020) | 37 | NIST staff author (Thomas); 17 U.S.C. 105 | https://nvlpubs.nist.gov/nistpubs/ams/NIST.AMS.200-9.pdf | 20b7ccfd54e4bacc57a43d47044fbd4cc35b068c424a825653cf4f3300719945 |
| nist-ams-100-37 | NIST AMS 100-37 Annual Report on U.S. Manufacturing Industry Statistics: 2020 | 60 | NIST staff author (Thomas); 17 U.S.C. 105 | https://nvlpubs.nist.gov/nistpubs/ams/NIST.AMS.100-37.pdf | b851392d5d5b214f8ca3a2a1d1c13dd9871c5f99191b63b809498e84fbc04889 |
