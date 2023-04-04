cd /global/cfs/cdirs/des/shivamp/cosmosis2p0/cosmosis-standard-library/
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp2_OmAsonly_MOPED.ini -p runtime.sampler=multinest
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp3_OmAsonly_MOPED.ini -p runtime.sampler=multinest
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp2kp3_autosm_gtsc.ini -p runtime.sampler=multinest
srun cosmosis --mpi gen_moments/cosmosis/ini/params_chain_kp2kp3_all_gtsc.ini -p runtime.sampler=multinest


cosmosis gen_moments/cosmosis/ini/params_chain_kp2kp3_autosm_gtsc.ini -p runtime.sampler=test