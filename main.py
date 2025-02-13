from project_utils import generate_log_ticks, symlog_transform

original_ticks = generate_log_ticks(vmin = -10000000000000000000,vmax = 10000000000000000000)
print(original_ticks)

transformed_ticks = symlog_transform(original_ticks)

print(transformed_ticks, [str(int(tick)) for tick in original_ticks])

