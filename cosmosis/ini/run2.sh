cd /global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp2_OmAsonly_MOPED2.ini -p runtime.sampler=multinest
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp3_OmAsonly_MOPED2.ini -p runtime.sampler=multinest
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp2_kp3_OmAsonly_MOPED2.ini -p runtime.sampler=multinest