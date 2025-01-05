import scipy.stats as stats

def anova_calculator(ss_tr, ss_e, num_groups, total_sample_size):
    # Degrees of freedom
    df_tr = num_groups - 1
    df_e = total_sample_size - num_groups
    df_total = total_sample_size - 1
    
    # Mean Squares
    ms_tr = ss_tr / df_tr
    ms_e = ss_e / df_e
    
    # F-statistic
    f_statistic = ms_tr / ms_e
    
    # Total Sum of Squares
    ss_total = ss_tr + ss_e

    # P-value
    p_value = 1 - stats.f.cdf(f_statistic, df_tr, df_e)

    # Print the ANOVA table
    print(f"{'Source of Variation':<20} {'SS':<10} {'df':<10} {'MS':<15} {'F':<10}")
    print("-" * 65)
    print(f"{'Treatment':<20} {ss_tr:<10.3f} {df_tr:<10} {ms_tr:<15.3f} {f_statistic:<10.3f}")
    print(f"{'Error':<20} {ss_e:<10.3f} {df_e:<10} {ms_e:<15.3f}")
    print(f"{'Total':<20} {ss_total:<10.3f} {df_total:<10}")
    print(f"\nP-value = Fcdf({f_statistic:.3f},1E99,{df_tr},{df_e}) = {p_value:.4f}")

ss_tr = 24.419     # Sum of Squares for Treatment (SSTr)
ss_e = 613.138     # Sum of Squares for Error (SSE)
num_groups = 4  # Number of groups (k)
total_sample_size = 222  # Total number of observations (N)

anova_calculator(ss_tr, ss_e, num_groups, total_sample_size)
