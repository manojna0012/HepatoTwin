"""Virtual-patient simulator for the in-silico evaluation of the twin (Proposal section 17.14).

The simulator plays the patient: it generates each virtual patient's true monthly state from
published trial evidence (data/evidence/simulator_parameters.csv). It never uses the Layer 2
model, so the twin is not graded against its own output.
"""
