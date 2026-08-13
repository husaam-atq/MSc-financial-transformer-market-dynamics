# Phase 6 Event-Level Evaluation Decision

The corrected test labels form 698 contiguous positive asset episodes. The Transformer detects 398 episode onsets (recall 0.5702) and at least one window in 0.5845 of episodes, with 3,372 false-positive windows and alert exposure of 0.2660.

A leave-one-event-out model experiment was not run. Contiguous label runs are label episodes, not independent economic events: many share dates and common shocks, and 60-session inputs overlap. Treating each run as an independent fold would overstate independence and repeatedly retrain after the historical test had been opened.

The static asset prior detects 0.5630 of onsets with 2,622 false-positive windows and 0.2331 alert exposure. Its reported zero false-alarm *episodes* is misleading because persistent alerts overlap positive runs; window counts and exposure are the safer controls. No defensible lead-time claim is available because the label is a forward 10-session window rather than an externally timestamped event catalogue.

An actual leave-one-event-out design requires an external event taxonomy, non-overlapping information windows, and new outcomes.
