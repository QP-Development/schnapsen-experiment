import os
import random
import pathlib
import shutil
import time

from typing import Optional

import click
from schnapsen.alternative_engines.ace_one_engine import AceOneGamePlayEngine

from schnapsen.bots import MLDataBot, train_ML_model, MLPlayingBot, RandBot, MiniMaxBot, SchnapsenServer

from schnapsen.bots.example_bot import ExampleBot

from schnapsen.game import (Bot, GamePlayEngine, Move, PlayerPerspective,
                            SchnapsenGamePlayEngine, TrumpExchange)
from schnapsen.alternative_engines.twenty_four_card_schnapsen import TwentyFourSchnapsenGamePlayEngine

from schnapsen.bots.rdeep import RdeepBot


@click.group()
def main() -> None:
    """Various Schnapsen Game Examples"""


def play_games_and_return_stats(engine: GamePlayEngine, bot1: Bot, bot2: Bot, number_of_games: int) -> int:
    """
    Play number_of_games games between bot1 and bot2, using the SchnapsenGamePlayEngine, and return how often bot1 won.
    Prints progress.
    """
    bot1_wins: int = 0
    lead, follower = bot1, bot2
    for i in range(1, number_of_games + 1):
        if i % 2 == 0:
            # swap bots so both start the same number of times
            lead, follower = follower, lead
        winner, _, _ = engine.play_game(lead, follower, random.Random(i))
        if winner == bot1:
            bot1_wins += 1
        if i % 500 == 0:
            print(f"Progress: {i}/{number_of_games}")
    return bot1_wins


@main.command()
def random_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RandBot(random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


class NotificationExampleBot(Bot):

    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        moves = perspective.valid_moves()
        return moves[0]

    def notify_game_end(self, won: bool, perspective: PlayerPerspective) -> None:
        print(f'result {"win" if won else "lost"}')
        print(f'I still have {len(perspective.get_hand())} cards left')

    def notify_trump_exchange(self, move: TrumpExchange) -> None:
        print(f"That trump exchanged! {move.jack}")


@main.command()
def notification_game() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = NotificationExampleBot()
    bot2 = RandBot(random.Random(464566))
    engine.play_game(bot1, bot2, random.Random(94))


class HistoryBot(Bot):
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move]) -> Move:
        history = perspective.get_game_history()
        print(f'the initial state of this game was {history[0][0]}')
        moves = perspective.valid_moves()
        return moves[0]


@main.command()
def try_example_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    bot1 = ExampleBot()
    bot2 = RandBot(random.Random(464566))
    winner, points, score = engine.play_game(bot1, bot2, random.Random(1))
    print(f"Winner is: {winner}, with {points} points, score {score}!")


@main.command()
def rdeep_game() -> None:
    bot1: Bot
    bot2: Bot
    engine = SchnapsenGamePlayEngine()
    rdeep = bot1 = RdeepBot(num_samples=16, depth=4, rand=random.Random(4564654644))
    bot2 = RandBot(random.Random(464566))
    wins = 0
    amount = 100
    for game_number in range(1, amount + 1):
        if game_number % 2 == 0:
            bot1, bot2 = bot2, bot1
        winner_id, _, _ = engine.play_game(bot1, bot2, random.Random(game_number))
        if winner_id == rdeep:
            wins += 1
        if game_number % 10 == 0:
            print(f"won {wins} out of {game_number}")


@main.group()
def ml() -> None:
    """Commands for the ML bot"""


@ml.command()
def create_replay_memory_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 10000
    replay_memory_dir: str = 'ML_replay_player_memories'
    replay_memory_filename: str = 'random_random_10k_games.txt'
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename

    bot_1_behaviour: Bot = RandBot(random.Random(5234243))
    # bot_1_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(4564654644))
    bot_2_behaviour: Bot = RandBot(random.Random(54354))
    # bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(68438))
    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=replay_memory_location)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=replay_memory_location)
    for i in range(1, num_of_games + 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{num_of_games}")
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")


# @ml.command()
def create_random_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 20
    replay_memory_dir: str = 'ML_generated_random_memories'
    replay_memory_filename: str = 'random_random_20_games' + time.time().__str__() + '.txt'
    print("creating dataset " + replay_memory_filename)
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename

    bot_1_behaviour: Bot = RandBot(random.Random(5234243))
    # bot_1_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(4564654644))
    bot_2_behaviour: Bot = RandBot(random.Random(54354))
    # bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(68438))
    delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
    if delete_existing_older_dataset and replay_memory_location.exists():
        print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
        replay_memory_location.unlink()

    # in any case make sure the directory exists
    replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

    # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
    engine = SchnapsenGamePlayEngine()
    replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=replay_memory_location)
    replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=replay_memory_location)
    for i in range(1, num_of_games + 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{num_of_games}")
        engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
    print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")

@ml.command()
def create_player_dataset() -> None:
    # define replay memory database creation parameters
    num_of_games: int = 20
    replay_memory_dir: str = 'ML_generated_player_memories'
    replay_memory_filename: str = 'random_player_20_games_' + time.time().__round__().__str__() + '.txt'
    replay_memory_location = pathlib.Path(replay_memory_dir) / replay_memory_filename


    with SchnapsenServer() as s:
        bot_1_behaviour: Bot = RandBot(random.Random(5234243))
        # bot_1_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(4564654644))
        bot_2_behaviour: Bot = s.make_gui_bot(name="randbot for training")
        # bot_2_behaviour: Bot = RdeepBot(num_samples=4, depth=4, rand=random.Random(68438))
        delete_existing_older_dataset = False

    # check if needed to delete any older versions of the dataset
        if delete_existing_older_dataset and replay_memory_location.exists():
            print(f"An existing dataset was found at location '{replay_memory_location}', which will be deleted as selected.")
            replay_memory_location.unlink()

        # in any case make sure the directory exists
        replay_memory_location.parent.mkdir(parents=True, exist_ok=True)

        # create new replay memory dataset, according to the behaviour of the provided bots and the provided random seed
        engine = SchnapsenGamePlayEngine()
        replay_memory_recording_bot_1 = MLDataBot(bot_1_behaviour, replay_memory_location=replay_memory_location)
        replay_memory_recording_bot_2 = MLDataBot(bot_2_behaviour, replay_memory_location=replay_memory_location)
        for i in range(1, num_of_games + 1):
            if i % 500 == 0:
                print(f"Progress: {i}/{num_of_games}")
            engine.play_game(replay_memory_recording_bot_1, replay_memory_recording_bot_2, random.Random(i))
        print(f"Replay memory dataset recorder for {num_of_games} games.\nDataset is stored at: {replay_memory_location}")

@ml.command()
def prep_random_experiment():
    try:
        shutil.rmtree(pathlib.Path("ML_generated_random_memories"))
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(pathlib.Path("ML_replay_random_memories"))
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(pathlib.Path("ML_random_models"))
    except FileNotFoundError:
        pass
    sets: int = 15
    print("creating datasets")
    for x in range(sets):
        create_random_dataset()
    shutil.copytree(pathlib.Path("ML_generated_random_memories"), pathlib.Path("ML_replay_random_memories"))
    print("combining datasets")
    combine_random_training_data()
    print("training models")
    train_random_models()
    # print("trying games")
    # try_random_games()

@ml.command()
def prep_player_experiment():
    try:
        shutil.rmtree(pathlib.Path("ML_replay_player_memories"))
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(pathlib.Path("ML_player_models"))
    except FileNotFoundError:
        pass
    shutil.copytree(pathlib.Path("ML_generated_player_memories"), pathlib.Path("ML_replay_player_memories"))
    combine_player_training_data()
    train_player_models()
    # try_player_games()



# @ml.command()
def combine_random_training_data() -> None:
    memoryloc = pathlib.Path("ML_replay_random_memories")
    files = os.listdir(memoryloc)
    for n in range(len(files) - 1):
        print("combining sets " + (n+1).__str__() +" and " + (n+2).__str__())
        combine(files, memoryloc, n) #DONT multithread this, it needs to be sequential to combine correctly
    for n in range(len(files)):
        os.rename(memoryloc / files[n], memoryloc / ((n+1).__str__() + '.txt'))


def combine(files, memoryloc, n):
    file1_path = files[n]
    file2_path = files[n + 1]
    with open(memoryloc / file1_path, 'r') as file1:
        with open(memoryloc / file2_path, 'a') as file2:
            shutil.copyfileobj(file1, file2)


# @ml.command()
def combine_player_training_data() -> None:
    memoryloc = pathlib.Path("ML_replay_player_memories")
    files = os.listdir(memoryloc)
    for n in range(len(files) - 1):
        combine(files, memoryloc, n)
    for n in range(len(files)):
        os.rename(memoryloc / files[n], memoryloc / ((n+1).__str__() + '.txt'))

@ml.command()
def train_model() -> None:
    # directory where the replay memory is saved
    replay_memory_filename: str = 'random_random_10k_games.txt'
    # filename of replay memory within that directory
    replay_memories_directory: str = 'ML_replay_player_memories'
    # Whether to train a complicated Neural Network model or a simple one.
    # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
    # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
    # under the code of body of the if statement 'if use_neural_network:'
    replay_memory_location = pathlib.Path(replay_memories_directory) / replay_memory_filename
    model_name: str = 'simple_model'
    model_dir: str = "ML_models"
    model_location = pathlib.Path(model_dir) / model_name
    overwrite: bool = False

    if overwrite and model_location.exists():
        print(f"Model at {model_location} exists already and will be overwritten as selected.")
        model_location.unlink()

    train_ML_model(replay_memory_location=replay_memory_location, model_location=model_location,
                   model_class='LR')

# @ml.command()
def train_random_models() -> None:
    replay_memories_directory: str = 'ML_replay_random_memories'
    rand_or_player: str = "random"
    # directory where the replay memory is saved
    for n in range(len(os.listdir(pathlib.Path(replay_memories_directory)))):
        train_random_experiment_model(n)


def train_random_experiment_model(n):

    replay_memories_directory: str = 'ML_replay_random_memories'
    rand_or_player: str = "random"
    # filename of replay memory within that directory
    replay_memory_filename: str = (n + 1).__str__() + '.txt'
    print("training model from " + replay_memory_filename)
    # Whether to train a complicated Neural Network model or a simple one.
    # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
    # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
    # under the code of body of the if statement 'if use_neural_network:'
    replay_memory_location = pathlib.Path(replay_memories_directory) / replay_memory_filename
    model_name: str = rand_or_player + '_model_' + (n + 1).__str__()
    model_dir: str = "ML_" + rand_or_player + "_models"
    model_location = pathlib.Path(model_dir) / model_name
    overwrite: bool = False
    if overwrite and model_location.exists():
        print(f"Model at {model_location} exists already and will be overwritten as selected.")
        model_location.unlink()
    train_ML_model(replay_memory_location=replay_memory_location, model_location=model_location,
                   model_class='LR')

def train_player_experiment_model(n):

    replay_memories_directory: str = 'ML_replay_player_memories'

    rand_or_player: str = "player"
    # filename of replay memory within that directory
    replay_memory_filename: str = (n + 1).__str__() + '.txt'
    # Whether to train a complicated Neural Network model or a simple one.
    # Tips: a neural network usually requires bigger datasets to be trained on, and to play with the parameters of the model.
    # Feel free to play with the hyperparameters of the model in file 'ml_bot.py', function 'train_ML_model',
    # under the code of body of the if statement 'if use_neural_network:'
    replay_memory_location = pathlib.Path(replay_memories_directory) / replay_memory_filename
    model_name: str = rand_or_player + '_model_' + (n + 1).__str__()
    model_dir: str = "ML_" + rand_or_player + "_models"
    model_location = pathlib.Path(model_dir) / model_name
    overwrite: bool = False
    if overwrite and model_location.exists():
        print(f"Model at {model_location} exists already and will be overwritten as selected.")
        model_location.unlink()
    train_ML_model(replay_memory_location=replay_memory_location, model_location=model_location,
                   model_class='LR')


# @ml.command()
def train_player_models() -> None:
    # directory where the replay memory is saved
    replay_memories_directory: str = 'ML_replay_player_memories'

    rand_or_player: str = "player"

    for x in range(len(os.listdir(pathlib.Path(replay_memories_directory)))):
        train_player_experiment_model(x)

@ml.command()
def try_bot_game() -> None:
    engine = SchnapsenGamePlayEngine()
    model_dir: str = 'ML_models'
    model_name: str = 'simple_model'
    model_location = pathlib.Path(model_dir) / model_name
    bot1: Bot = MLPlayingBot(model_location=model_location)
    bot2: Bot = MiniMaxBot()
    number_of_games: int = 10000

    # play games with altering leader position on first rounds
    ml_bot_wins_against_random = play_games_and_return_stats(engine=engine, bot1=bot1, bot2=bot2, number_of_games=number_of_games)
    print(f"The ML bot with name {model_name}, won {ml_bot_wins_against_random} times out of {number_of_games} games played.")

def try_random_games() -> None:
    model_dir: str = 'ML_random_models'
    for n in range(len(os.listdir(pathlib.Path(model_dir)))):
        play_random_game(n)


@ml.command()
def try_player_game_1() -> None:
    play_player_game(1)


@ml.command()
def try_player_game_2() -> None:
    play_player_game(2)


@ml.command()
def try_player_game_3() -> None:
    play_player_game(3)


@ml.command()
def try_player_game_4() -> None:
    play_player_game(4)


@ml.command()
def try_player_game_5() -> None:
    play_player_game(5)


@ml.command()
def try_player_game_6() -> None:
    play_player_game(6)


@ml.command()
def try_player_game_7() -> None:
    play_player_game(7)


@ml.command()
def try_player_game_8() -> None:
    play_player_game(8)


@ml.command()
def try_player_game_9() -> None:
    play_player_game(9)


@ml.command()
def try_player_game_10() -> None:
    play_player_game(10)


@ml.command()
def try_player_game_11() -> None:
    play_player_game(11)


@ml.command()
def try_player_game_12() -> None:
    play_player_game(12)


@ml.command()
def try_player_game_13() -> None:
    play_player_game(13)


@ml.command()
def try_player_game_14() -> None:
    play_player_game(14)


@ml.command()
def try_player_game_0() -> None:
    play_player_game(0)

@ml.command()
def try_random_game_1() -> None:
    play_random_game(1)


@ml.command()
def try_random_game_2() -> None:
    play_random_game(2)


@ml.command()
def try_random_game_3() -> None:
    play_random_game(3)


@ml.command()
def try_random_game_4() -> None:
    play_random_game(4)


@ml.command()
def try_random_game_5() -> None:
    play_random_game(5)


@ml.command()
def try_random_game_6() -> None:
    play_random_game(6)


@ml.command()
def try_random_game_7() -> None:
    play_random_game(7)


@ml.command()
def try_random_game_8() -> None:
    play_random_game(8)


@ml.command()
def try_random_game_9() -> None:
    play_random_game(9)


@ml.command()
def try_random_game_10() -> None:
    play_random_game(10)


@ml.command()
def try_random_game_11() -> None:
    play_random_game(11)


@ml.command()
def try_random_game_12() -> None:
    play_random_game(12)


@ml.command()
def try_random_game_13() -> None:
    play_random_game(13)


@ml.command()
def try_random_game_14() -> None:
    play_random_game(14)


@ml.command()
def try_random_game_0() -> None:
    play_random_game(0)


def play_random_game(n):
    print("playing game: " + (n+1).__str__())
    engine = SchnapsenGamePlayEngine()
    model_dir: str = 'ML_random_models'
    rand_or_player: str = "random"
    # filename of replay memory within that directory
    replay_memory_filename: str = (n + 1).__str__() + '.txt'
    model_name: str = rand_or_player + '_model_' + (n + 1).__str__()
    model_location = pathlib.Path(model_dir) / model_name
    bot1: Bot = MLPlayingBot(model_location=model_location)
    # bot2: Bot = RandBot(random.Random(78465))
    bot2: Bot = RdeepBot(rand=random.Random(78465), depth=4, num_samples=20)
    number_of_games: int = 10000
    # play games with altering leader position on first rounds
    ml_bot_wins_against_random = play_games_and_return_stats(engine=engine, bot1=bot1, bot2=bot2,
                                                             number_of_games=number_of_games)
    print(
        f"The ML bot with name {model_name}, trained on {(n + 1) * 20} {rand_or_player} games, won {ml_bot_wins_against_random} times out of {number_of_games} games played.")

def play_player_game(n):
    engine = SchnapsenGamePlayEngine()
    model_dir: str = 'ML_player_models'
    rand_or_player: str = "player"
    # filename of replay memory within that directory
    replay_memory_filename: str = (n + 1).__str__() + '.txt'
    model_name: str = rand_or_player + '_model_' + (n + 1).__str__()
    model_location = pathlib.Path(model_dir) / model_name
    bot1: Bot = MLPlayingBot(model_location=model_location)
    # bot2: Bot = RandBot(random.Random(78465))
    bot2: Bot = RdeepBot(rand=random.Random(78465), depth=4, num_samples=20)
    number_of_games: int = 10000
    # play games with altering leader position on first rounds
    ml_bot_wins_against_random = play_games_and_return_stats(engine=engine, bot1=bot1, bot2=bot2,
                                                             number_of_games=number_of_games)
    print(
        f"The ML bot with name {model_name}, trained on {(n + 1) * 20} {rand_or_player} games, won {ml_bot_wins_against_random} times out of {number_of_games} games played.")


def try_player_games() -> None:
    model_dir: str = 'ML_player_models'

    for x in range(len(os.listdir(pathlib.Path(model_dir)))):
        play_player_game(x)

@main.command()
def game_24() -> None:
    engine = TwentyFourSchnapsenGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RandBot(random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


@main.command()
def game_ace_one() -> None:
    engine = AceOneGamePlayEngine()
    bot1 = RandBot(random.Random(12112121))
    bot2 = RdeepBot(num_samples=16, depth=4, rand=random.Random(464566))
    for i in range(1000):
        winner_id, game_points, score = engine.play_game(bot1, bot2, random.Random(i))
        print(f"Game ended. Winner is {winner_id} with {game_points} points, score {score}")


if __name__ == "__main__":
    main()
