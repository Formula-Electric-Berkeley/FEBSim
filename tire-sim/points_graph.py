import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

events = ['Cost', 'Presentation', 'Design', 'Acceleration', 'Skid Pad', 'Autocross', 'Endurance', 'Efficiency']
# events = list(reversed(events))

# Top 10 teams : these have labels
labeled_teams = {
    'Oregon State':             [75.2, 18.9, 120.0,  84.0,  75.0, 120.3, 248.4, 48.7],
    'San Jose State':           [86.6, 63.4, 115.0,  71.5,  53.9, 119.9, 275.0,  0.0],
    'Georgia Tech':             [82.1, 15.0, 135.0, 100.0,  72.8, 114.7, 187.3, 65.2],
    'Ecole Polytechnique Mtl':  [78.6, 26.0, 110.0,  76.7,  57.9, 123.1, 250.5, 33.2],
    'Univ of Washington':       [76.8, 59.7, 105.0,  95.6,  63.6, 125.0, 166.7, 61.2],
    'Univ of Pittsburgh':       [81.3, 67.9, 100.0,  58.7,  45.4, 102.9, 195.0, 72.2],
    'Natl Univ of Singapore':   [73.4, 64.3, 125.0,  69.8,  64.3, 113.1, 127.4, 71.3],
    'RIT':                      [72.1, 75.0, 100.0,  99.9,  74.4,  43.2, 166.1, 71.0],
    'MIT':                      [72.2, 58.5, 110.0,  69.5,  45.3, 106.5,  72.1, 63.6],
    'Purdue':                   [86.4, 17.3, 105.0,  24.8,  54.7, 108.2, 140.5, 56.0],
}

# labeled_teams = {k : reversed(v) for (k, v) in labeled_teams.items()}

# unlabeled taeams
background_teams = [
    [78.8, 54.5, 125.0, 32.5, 74.8, 113.1,  23.0, 90.9],  # Cal Poly SLO
    [82.0, 60.1, 150.0, 75.4, 73.4, 123.7,   2.0,  0.0],  # Wisconsin
    [90.2, 50.0, 130.0, 81.3, 56.4,  89.0,  21.0,  0.0],  # Carnegie Mellon
    [63.3, 18.9,  70.0,102.3,  0.0, 152.9, 100.0,  0.0],  # UConn
    [72.9, 60.6, 129.0, 59.1, 47.0,  97.2,  25.0,  0.0],  # Toronto
    [42.7, 19.6,  65.0, 54.2, 39.8,  84.6, 120.2, 57.2],  # Cincinnati
    [72.8, 44.1, 120.0, 48.7, 50.6, 110.2,  14.0,  0.0],  # Kookmin
    [42.2, 51.5,  75.0, 48.8, 44.1,   6.5, 118.8, 76.6],  # UC Santa Cruz
    [87.1, 67.3,  95.0, 19.5, 49.9,  79.2,  25.0,  0.0],  # Laval
    [75.3, 65.0, 115.0, 96.5, 36.9,   2.0,   0.0,  0.0],  # Michigan
    [58.5, 57.6,  65.0, 46.8, 53.0, 104.4,  14.0,  0.0],  # Penn State
    [77.7, 54.8, 125.0, 30.4, 28.1,  22.0,   0.0,  0.0],  # Cornell
    [74.3, 16.2,  85.0, 44.4, 32.7,  59.2,   2.0,  0.0],  # Northwestern
    [74.1, 20.0,  65.0,  4.5, 19.5,  59.7,  60.5,  0.0],  # Alberta
    [73.6, 40.4, 105.0, 24.0, 55.6,   0.0,   0.0,  0.0],  # Florida
    [90.4, 74.3, 120.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # UPenn
    [66.4, 66.0,  95.0,  6.3,  3.5,   6.5,  14.0,  0.0],  # Waterloo
    [47.0, 32.2,  70.0,  4.5, 43.5,  50.1,  25.0,  0.0],  # Auburn
    [68.9, 68.3, 115.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # UIUC
    [67.4, 67.9, 100.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # Texas A&M
    [70.8, 56.5, 100.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # Columbia
    [87.7, 16.0, 115.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # McGill
    [81.2, 40.3,  90.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # UC Davis
    [70.7, 64.5,  75.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # Michigan State
    [84.1, 46.7,  95.0,  0.0,  0.0,   0.0,   0.0,  0.0],  # UCLA
]

# background_teams = [reversed(bt) for bt in background_teams]

sn5_goals = [85.0, 70.0, 120.0, 54.1, 32.67, 86.45, 129.05, 15.85]

# sn5_goals = list(reversed(sn5_goals))

def cumulative(scores):
    total = 0
    cum = []
    for s in scores:
        total += s
        cum.append(total)
    return cum

colors = [
    '#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261',
    '#264653', '#a8dadc', '#6d6875', '#b5838d', '#52b788',
]

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x = np.arange(len(events))

# unlabeled
for scores in background_teams:
    ax.plot(x, cumulative(scores), color='#cccccc', linewidth=1.0, alpha=0.6, zorder=1)

# labeled
for i, (team, scores) in enumerate(labeled_teams.items()):
    ax.plot(x, cumulative(scores), color=colors[i], linewidth=1.8, label=team, alpha=0.6, zorder=2)

# our goals
ax.plot(x, cumulative(sn5_goals), color='#003262', linewidth=3.5,
        linestyle='-', marker='o', markersize=7, markerfacecolor='#FDB515',
        label='UC Berkeley (SN5 Goals)', zorder=10)

ax.set_xticks(x)
ax.set_xticklabels(events, fontsize=13)
ax.set_ylabel('Cumulative Points', fontsize=13)
ax.set_title('Event Points by School (2025 Results, FEB 2026 Goals)', fontsize=16, pad=16, fontweight='bold')

ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=11, frameon=False)

plt.tight_layout()
plt.savefig('fsae_chart.png', dpi=180, bbox_inches='tight')
plt.show()
