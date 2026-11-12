# Independent competition-output QC

**Passed integrity and arithmetic checks.** Original predictions and receipts were not modified.

- Reconstructed all 309 predictions independently with Decimal arithmetic and weights 5:1:1:1:1:1; every rounded value matches. IDs, order, finite values and inherited ranges pass.
- Verified all 10 original output hashes, six source/protocol hashes, 10 input hashes and inherited checkpoint records.
- Confirmed 299 changed rows and 10 all-new-head failures. Every fallback triplet preserves the prior CSV values exactly. Three images have one surviving head.
- Confirmed unchanged per-head MT, original grouping, cached texture and prior validity flags; final MT changes arise from changed head weights.
- Closed the prior-receipt discrepancy: the retrieved Kaggle receipt matches the build's recorded SHA-256. Its JSON keys and values equal the local finalized receipt; only the insertion position of `one_valid_head_images` differs.

Candidate SHA-256: `91884e84dd84aa632a226c3938be1fcc1e5147e061c3d26ca75893a76c99bc76`.

The 1,854-row map manifest has complete unique paths and valid hashes. Raw map bytes were not included in the local download and were not independently rechecked here. This audit does not assess predictive accuracy or verify leaderboard results.
