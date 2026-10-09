import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('tottime').print_stats(30)
