# Third-party evaluation executable

`orca/orca` is the prebuilt Linux ORCA graphlet-orbit counter used by the
exact CatFlow and DiGress/SPECTRE orbit metrics. It is third-party evaluation
software and is not part of the submitted model implementation.

The launchers record its SHA-256 digest in generated metric reports so that a
replacement build cannot silently change an experiment.
