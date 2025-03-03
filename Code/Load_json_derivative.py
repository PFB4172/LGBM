df_from_json_pandas = pd.read_json('周度风险_趋势.json')
df_from_json_pandas2 = pd.read_json('周度风险_比率.json')
df_from_json_pandas3 = pd.read_json('标品趋势.json')
df_from_json_pandas4 = pd.read_json('标品比率.json')

commands = pd.concat([df_from_json_pandas['衍生python代码'],
                      df_from_json_pandas2['衍生python代码'],
                     df_from_json_pandas3['衍生python代码'],
                     df_from_json_pandas4['衍生python代码']],
                     axis=0,ignore_index=True)
for cmd in commands:
    try:
        exec(cmd)
    except Exception as e:
        print(f"Error executing command: {e}")