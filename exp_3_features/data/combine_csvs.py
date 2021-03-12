import pandas as pd     
def combine_csvs(fnames_in, fname_out):
    combined_csv = pd.concat([pd.read_csv(f).assign(pid= pid_type) for f, pid_type in fnames_in ])
    combined_csv.to_csv( fname_out, index=False)

if __name__=="__main__":
    #combine_csvs([("test1.csv", "e"), ("test2.csv", "mu")], "test_combined.csv") 
    
    """
    combine_csvs([
        ("NTuple_BKstee_516_517_VAE_K_in.csv", "k"),
        ("NTuple_BKstee_516_517_VAE_l1_in.csv", "e"),
        ("NTuple_BKstee_516_517_VAE_l2_in.csv", "e"),
        ("NTuple_BKstmumu_518_519_VAE_K_in.csv", "k"),
        ("NTuple_BKstmumu_518_519_VAE_l1_in.csv", "m"),
        ("NTuple_BKstmumu_518_519_VAE_l2_in.csv", "m"),
        ], "data_for_gan_combined_origtrain.csv")
    """

    combine_csvs([
        ("NTuple_BKee_481_482_994962_to_-1_VAE_K_in.csv", "k"),
        ("NTuple_BKee_481_482_994962_to_-1_VAE_l1_in.csv", "e"),
        ("NTuple_BKee_481_482_994962_to_-1_VAE_l2_in.csv", "e"),
        ("NTuple_BKmumu_483_484_1000322_to_-1_VAE_K_in.csv", "k"),
        ("NTuple_BKmumu_483_484_1000322_to_-1_VAE_l1_in.csv", "m"),
        ("NTuple_BKmumu_483_484_1000322_to_-1_VAE_l2_in.csv", "m"),
        ], "data_for_gan_combined_ktrain.csv")

    combine_csvs([
        ("NTuple_BKstee_516_517_676942_to_-1_VAE_K_in.csv", "k"),
        ("NTuple_BKstee_516_517_676942_to_-1_VAE_l1_in.csv", "e"),
        ("NTuple_BKstee_516_517_676942_to_-1_VAE_l2_in.csv", "e"),
        ("NTuple_BKstmumu_518_519_591567_to_-1_VAE_K_in.csv", "k"),
        ("NTuple_BKstmumu_518_519_591567_to_-1_VAE_l1_in.csv", "m"),
        ("NTuple_BKstmumu_518_519_591567_to_-1_VAE_l2_in.csv", "m"),
        ], "data_for_gan_combined_ksttrain.csv")
    
    combine_csvs([
        ("NTuple_BKee_481_482_994962_to_-1_VAE_K_in_spdrestr.csv", "k"),
        ("NTuple_BKee_481_482_994962_to_-1_VAE_l1_in_spdrestr.csv", "e"),
        ("NTuple_BKee_481_482_994962_to_-1_VAE_l2_in_spdrestr.csv", "e"),
        ("NTuple_BKmumu_483_484_1000322_to_-1_VAE_K_in_spdrestr.csv", "k"),
        ("NTuple_BKmumu_483_484_1000322_to_-1_VAE_l1_in_spdrestr.csv", "m"),
        ("NTuple_BKmumu_483_484_1000322_to_-1_VAE_l2_in_spdrestr.csv", "m"),
        ], "data_for_gan_combined_ktrain_spdrestr.csv")

    combine_csvs([
        ("NTuple_BKstee_516_517_676942_to_-1_VAE_K_in_spdrestr.csv", "k"),
        ("NTuple_BKstee_516_517_676942_to_-1_VAE_l1_in_spdrestr.csv", "e"),
        ("NTuple_BKstee_516_517_676942_to_-1_VAE_l2_in_spdrestr.csv", "e"),
        ("NTuple_BKstmumu_518_519_591567_to_-1_VAE_K_in_spdrestr.csv", "k"),
        ("NTuple_BKstmumu_518_519_591567_to_-1_VAE_l1_in_spdrestr.csv", "m"),
        ("NTuple_BKstmumu_518_519_591567_to_-1_VAE_l2_in_spdrestr.csv", "m"),
        ], "data_for_gan_combined_ksttrain_spdrestr.csv")

