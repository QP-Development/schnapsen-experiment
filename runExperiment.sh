echo "make sure to set model_class on line 343 and 365 in executables/cli.py to 'NN' or 'LR' for a Neural Network or Logistic Regression based ML-bot respectively"
python executables/cli.py ml prep-player-experiment > prep-player.txt
python executables/cli.py ml prep-random-experiment > prep-random.txt
#run all games in parallel to wait 1/12th of the time (on 12-threaded processor)
(python executables/cli.py ml try-player-game-0 > output_player_0.txt) &
(python executables/cli.py ml try-player-game-1 > output_player_1.txt) &
(python executables/cli.py ml try-player-game-2 > output_player_2.txt) &
(python executables/cli.py ml try-player-game-3 > output_player_3.txt) &
(python executables/cli.py ml try-player-game-4 > output_player_4.txt) &
(python executables/cli.py ml try-player-game-5 > output_player_5.txt) &
(python executables/cli.py ml try-player-game-6 > output_player_6.txt) &
(python executables/cli.py ml try-player-game-7 > output_player_7.txt) &
(python executables/cli.py ml try-player-game-8 > output_player_8.txt) &
(python executables/cli.py ml try-player-game-9 > output_player_9.txt) &
(python executables/cli.py ml try-player-game-10> output_player_10.txt) &
(python executables/cli.py ml try-player-game-11> output_player_11.txt) &
(python executables/cli.py ml try-player-game-12> output_player_12.txt) &
(python executables/cli.py ml try-player-game-13> output_player_13.txt) &
(python executables/cli.py ml try-player-game-14> output_player_14.txt)&
(python executables/cli.py ml try-random-game-0 > output_random_0.txt) &
(python executables/cli.py ml try-random-game-1 > output_random_1.txt) &
(python executables/cli.py ml try-random-game-2 > output_random_2.txt) &
(python executables/cli.py ml try-random-game-3 > output_random_3.txt) &
(python executables/cli.py ml try-random-game-4 > output_random_4.txt) &
(python executables/cli.py ml try-random-game-5 > output_random_5.txt) &
(python executables/cli.py ml try-random-game-6 > output_random_6.txt) &
(python executables/cli.py ml try-random-game-7 > output_random_7.txt) &
(python executables/cli.py ml try-random-game-8 > output_random_8.txt) &
(python executables/cli.py ml try-random-game-9 > output_random_9.txt) &
(python executables/cli.py ml try-random-game-10> output_random_10.txt) &
(python executables/cli.py ml try-random-game-11> output_random_11.txt) &
(python executables/cli.py ml try-random-game-12> output_random_12.txt) &
(python executables/cli.py ml try-random-game-13> output_random_13.txt) &
(python executables/cli.py ml try-random-game-14> output_random_14.txt)