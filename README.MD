# SAMMiCA Scenario 3 AI Soccer
Source codes for AIWC multi agent reinforcement learning.

AIWC is a 5:5 robot soccer game where each team is comprised of one goalkeeper, two defenders, and two forwards.

| Robot id | Positions |
| -------- | --------- |
| robot_0 | Goalkeeper(GK)  |
| robot_1 | Defender #1(D1) |
| robot_2 | Defender #2(D2) |
| robot_3 | Forward #1(F1) |
| robot_4 | Forward #2(F2) |

## Dataset download

[AIWC_500_games.zip](https://kaistackr-my.sharepoint.com/:u:/g/personal/gm_lee_kaist_ac_kr/EeSrUXT_azlGrxnOMap_wJABzyhJJcadqHcgCt-jqeRAew?e=gitUI8)

### Format

Data is stored in .csv format with the following fields:

| index | field name | Data Type | Description |
|--- | ---------- | --------- | ------- |
| 0  | frame      | int       | Frame number starting from  0 |
| 1  | game_state | int       | 0:Default, 1:Kick-off, 2:Goal-kick, 3:Corner-kick, 4:Penalty-kick |
| 2  | friction   | float     | frictional coefficients of fields. 0.1:very slippery, 0.5:slippery, 3.0:default |
| 3  | ball_pos_0 | float     | ball's location on the x-axis(length of the field) |
| 4  | ball_pos_1 | float     | ball's location on the y-axis(width of the field) |
| 5  | ball_pos_2 | float     | ball's location on the z-axis(height above the court) |
| 6  | robot_0_wheel_speed_0 | float     | Linear wheel velocity of GK's left wheel |
| 7  | robot_0_wheel_speed_1 | float     | Linear wheel velocity of GK's right wheel |
| 8  | robot_1_wheel_speed_0 | float     | Linear wheel velocity of D1's left wheel |
| 9  | robot_1_wheel_speed_1 | float     | Linear wheel velocity of D1's right wheel |
| 10 | robot_2_wheel_speed_0 | float     | Linear wheel velocity of D2's left wheel |
| 11 | robot_2_wheel_speed_1 | float     | Linear wheel velocity of D2's right wheel |
| 12 | robot_3_wheel_speed_0 | float     | Linear wheel velocity of F1's left wheel |
| 13 | robot_3_wheel_speed_1 | float     | Linear wheel velocity of F1's right wheel |
| 14 | robot_4_wheel_speed_0 | float     | Linear wheel velocity of F2's left wheel |
| 15 | robot_4_wheel_speed_1 | float     | Linear wheel velocity of F2's right wheel |
| 16 | robot_0_cur_pos_0 | float     | GK's location on the x-axis(length of the field)        |
| 17 | robot_0_cur_pos_1 | float     | GK's location on the y-axis(width of the field)        |
| 18 | robot_0_cur_pos_2 | float     | GK's location on the z-axis(height above the court)        |
| 19 | robot_1_cur_pos_0 | float     | D1's location on the x-axis(length of the field)        |
| 20 | robot_1_cur_pos_1 | float     | D1's location on the y-axis(width of the field)        |
| 21 | robot_1_cur_pos_2 | float     | D1's location on the z-axis(height above the court)        |
| 22 | robot_2_cur_pos_0 | float     | D2's location on the x-axis(length of the field)        |
| 23 | robot_2_cur_pos_1 | float     | D2's location on the y-axis(width of the field)        |
| 24 | robot_2_cur_pos_2 | float     | D2's location on the z-axis(height above the court)        |
| 25 | robot_3_cur_pos_0 | float     | F1's location on the x-axis(length of the field)        |
| 26 | robot_3_cur_pos_1 | float     | F1's location on the y-axis(width of the field)        |
| 27 | robot_3_cur_pos_2 | float     | F1's location on the z-axis(height above the court)        |
| 28 | robot_4_cur_pos_0 | float     | F2's location on the x-axis(length of the field)        |
| 29 | robot_4_cur_pos_1 | float     | F2's location on the y-axis(width of the field)        |
| 30 | robot_4_cur_pos_2 | float     | F2's location on the z-axis(height above the court)        |
| 31 | robot_0_cur_pos_opp_0 | float     | Opponent GK's location on the x-axis(length of the field)        |
| 32 | robot_0_cur_pos_opp_1 | float     | Opponent GK's location on the y-axis(width of the field)        |
| 33 | robot_0_cur_pos_opp_2 | float     | Opponent GK's location on the z-axis(height above the court)        |
| 34 | robot_1_cur_pos_opp_0 | float     | Opponent D1's location on the x-axis(length of the field)        |
| 35 | robot_1_cur_pos_opp_1 | float     | Opponent D1's location on the y-axis(width of the field)        |
| 36 | robot_1_cur_pos_opp_2 | float     | Opponent D1's location on the z-axis(height above the court)        |
| 37 | robot_2_cur_pos_opp_0 | float     | Opponent D2's location on the x-axis(length of the field)        |
| 38 | robot_2_cur_pos_opp_1 | float     | Opponent D2's location on the y-axis(width of the field)        |
| 39 | robot_2_cur_pos_opp_2 | float     | Opponent D2's location on the z-axis(height above the court)        |
| 40 | robot_3_cur_pos_opp_0 | float     | Opponent F1's location on the x-axis(length of the field)        |
| 41 | robot_3_cur_pos_opp_1 | float     | Opponent F1's location on the y-axis(width of the field)        |
| 42 | robot_3_cur_pos_opp_2 | float     | Opponent F1's location on the z-axis(height above the court)        |
| 43 | robot_4_cur_pos_opp_0 | float     | Opponent F2's location on the x-axis(length of the field)        |
| 44 | robot_4_cur_pos_opp_1 | float     | Opponent F2's location on the y-axis(width of the field)        |
| 45 | robot_4_cur_pos_opp_2 | float     | Opponent F2's location on the z-axis(height above the court)        |
| 46 | robot_0_active | boolean     | GK's activation state(True:active, False:disable)        |
| 47 | robot_1_active | boolean     | D1's activation state(True:active, False:disable)        |
| 48 | robot_2_active | boolean     | D2's activation state(True:active, False:disable)        |
| 49 | robot_3_active | boolean     | F1's activation state(True:active, False:disable)        |
| 50 | robot_4_active | boolean     | F2's activation state(True:active, False:disable)        |
| 51 | robot_0_active_opp | boolean     | Opponent GK's activation state(True:active, False:disable)        |
| 52 | robot_1_active_opp | boolean     | Opponent D1's activation state(True:active, False:disable)        |
| 53 | robot_2_active_opp | boolean     | Opponent D2's activation state(True:active, False:disable)        |
| 54 | robot_3_active_opp | boolean     | Opponent F1's activation state(True:active, False:disable)        |
| 55 | robot_4_active_opp | boolean     | Opponent F2's activation state(True:active, False:disable)        |
| 56 | episode_id | int     | Labels for each game_state sequence with consecutive frames |
