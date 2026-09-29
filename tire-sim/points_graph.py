import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

events = ['Cost', 'Presentation', 'Design', 'Acceleration', 'Skid Pad', 'Autocross', 'Endurance', 'Efficiency']

# Top 10 teams; these have labels
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

sn5_goals = [85.0, 70.0, 120.0, 54.1, 32.67, 86.45, 129.05, 15.85]

def get_long_df(team_dict, list_of_lists, goals):
    rows = []
    
    # Process Labeled Teams
    for name, scores in team_dict.items():
        cum_points = np.cumsum(scores)
        for event, points in zip(events, cum_points):
            rows.append({'Team': name, 'Event': event, 'Points': points, 'Type': 'Competitor'})
            
    # Process Background Teams
    for i, scores in enumerate(list_of_lists):
        cum_points = np.cumsum(scores)
        for event, points in zip(events, cum_points):
            rows.append({'Team': f'Background_{i}', 'Event': event, 'Points': points, 'Type': 'Competitor'})
            
    # Process Berkeley Goals
    cum_goals = np.cumsum(goals)
    for event, points in zip(events, cum_goals):
        rows.append({'Team': 'UC Berkeley', 'Event': event, 'Points': points, 'Type': 'Goal'})
        
    return pd.DataFrame(rows)

df = get_long_df(labeled_teams, background_teams, sn5_goals)

color_map = {
    'Oregon State': '#e63946', 'San Jose State': '#457b9d', 
    'Georgia Tech': '#2a9d8f', 'Ecole Polytechnique Mtl': '#e9c46a',
    'Univ of Washington': '#f4a261', 'Univ of Pittsburgh': '#264653',
    'Natl Univ of Singapore': '#a8dadc', 'RIT': '#6d6875', 
    'MIT': '#b5838d', 'Purdue': '#52b788'
}

plt.figure(figsize=(14, 8))

# unlabeled schools
df_background = df[~df['Team'].isin(color_map.keys()) & (df['Type'] == 'Competitor')]
sns.stripplot(
    data=df_background, x='Event', y='Points', 
    color='#dddddd', alpha=0.9, jitter=0.1, size=8, zorder=1, label="Other Schools"
)

# labeled schools
df_top10 = df[df['Team'].isin(color_map.keys())]
for team, team_color in color_map.items():
    team_data = df_top10[df_top10['Team'] == team]
    plt.scatter(
        x=team_data['Event'], y=team_data['Points'], 
        color=team_color, s=40, label=team, alpha=0.9, zorder=5
    )

# berkeley
sns.pointplot(
    data=df[df['Team'] == 'UC Berkeley'], 
    x='Event', y='Points', 
    color='#003262', markers='o', scale=1.5, linestyles='-', zorder=10, label="UC Berkeley (SN5 Goals)"
)
sns.scatterplot(data=df[df['Team'] == 'UC Berkeley'], x='Event', y='Points', color='#FDB515', s=80, zorder=11, edgecolors='#003262')

# just to make the labels be in order
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = ['UC Berkeley (SN5 Goals)'] + list(color_map.keys()) + ['Other Schools']
ordered_handles = [by_label[label] for label in ordered_labels if label in by_label]
plt.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.title('Event Points by School (2025 Results, FEB 2026 Goals)', fontsize=16, fontweight='bold')
plt.ylabel('Cumulative Points', fontsize=14)
sns.despine()
plt.subplots_adjust(right=0.75)
plt.savefig('fsae_chart.png', dpi=180, bbox_inches='tight')
plt.show()
