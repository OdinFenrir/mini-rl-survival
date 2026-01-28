"""Minimal viz compatibility module kept for scripts that expect viz.render_color."""

def render_color(env):
	# Fallback: return a simple ASCII representation
	rows = []
	for y in range(env.height):
		row = []
		for x in range(env.width):
			ch = "."
			if (x, y) == getattr(env, "goal", (None, None)):
				ch = "G"
			if (x, y) in getattr(env, "walls", set()):
				ch = "#"
			if (x, y) in getattr(env, "hazards", set()):
				ch = "T"
			if (x, y) == getattr(env, "food", (None, None)):
				ch = "F"
			if (x, y) == getattr(env, "agent", (None, None)):
				ch = "A"
			row.append(ch)
		rows.append("".join(row))
	return "\n".join(rows)
