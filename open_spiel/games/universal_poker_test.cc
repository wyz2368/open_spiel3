// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/universal_poker.h"

#include <memory>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/canonical_game_strings.h"
#include "open_spiel/algorithms/evaluate_bots.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace universal_poker {
namespace {

namespace testing = open_spiel::testing;

constexpr absl::string_view kKuhnLimit3P =
    ("GAMEDEF\n"
     "limit\n"
     "numPlayers = 3\n"
     "numRounds = 1\n"
     "blind = 1 1 1\n"
     "raiseSize = 1\n"
     "firstPlayer = 1\n"
     "maxRaises = 1\n"
     "numSuits = 1\n"
     "numRanks = 4\n"
     "numHoleCards = 1\n"
     "numBoardCards = 0\n"
     "END GAMEDEF\n");
GameParameters KuhnLimit3PParameters() {
  return {{"betting", GameParameter(std::string("limit"))},
          {"numPlayers", GameParameter(3)},
          {"numRounds", GameParameter(1)},
          {"blind", GameParameter(std::string("1 1 1"))},
          {"raiseSize", GameParameter(std::string("1"))},
          {"firstPlayer", GameParameter(std::string("1"))},
          {"maxRaises", GameParameter(std::string("1"))},
          {"numSuits", GameParameter(1)},
          {"numRanks", GameParameter(4)},
          {"numHoleCards", GameParameter(1)},
          {"numBoardCards", GameParameter(std::string("0"))}};
}

constexpr absl::string_view kHoldemNoLimit6P =
    ("GAMEDEF\n"
     "nolimit\n"
     "numPlayers = 6\n"
     "numRounds = 4\n"
     "stack = 20000 20000 20000 20000 20000 20000\n"
     "blind = 50 100 0 0 0 0\n"
     "firstPlayer = 3 1 1 1\n"
     "numSuits = 4\n"
     "numRanks = 13\n"
     "numHoleCards = 2\n"
     "numBoardCards = 0 3 1 1\n"
     "END GAMEDEF\n");
GameParameters HoldemNoLimit6PParameters() {
  return {{"betting", GameParameter(std::string("nolimit"))},
          {"numPlayers", GameParameter(6)},
          {"numRounds", GameParameter(4)},
          {"stack",
           GameParameter(std::string("20000 20000 20000 20000 20000 20000"))},
          {"blind", GameParameter(std::string("50 100 0 0 0 0"))},
          {"firstPlayer", GameParameter(std::string("3 1 1 1"))},
          {"numSuits", GameParameter(4)},
          {"numRanks", GameParameter(13)},
          {"numHoleCards", GameParameter(2)},
          {"numBoardCards", GameParameter(std::string("0 3 1 1"))}};
}

void LoadKuhnLimitWithAndWithoutGameDef() {
  UniversalPokerGame kuhn_limit_3p_gamedef(
      {{"gamedef", GameParameter(std::string(kKuhnLimit3P))}});
  UniversalPokerGame kuhn_limit_3p(KuhnLimit3PParameters());

  SPIEL_CHECK_EQ(kuhn_limit_3p_gamedef.GetACPCGame()->ToString(),
                 kuhn_limit_3p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(kuhn_limit_3p_gamedef.GetACPCGame())) ==
                   (*(kuhn_limit_3p.GetACPCGame())));
}

void LoadHoldemNoLimit6PWithAndWithoutGameDef() {
  UniversalPokerGame holdem_no_limit_6p_gamedef(
      {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  UniversalPokerGame holdem_no_limit_6p(HoldemNoLimit6PParameters());

  SPIEL_CHECK_EQ(holdem_no_limit_6p_gamedef.GetACPCGame()->ToString(),
                 holdem_no_limit_6p.GetACPCGame()->ToString());
  SPIEL_CHECK_TRUE((*(holdem_no_limit_6p_gamedef.GetACPCGame())) ==
                   (*(holdem_no_limit_6p.GetACPCGame())));
}
void LoadGameFromDefaultConfig() { LoadGame("universal_poker"); }

void LoadAndRunGamesFullParameters() {
  std::shared_ptr<const Game> kuhn_limit_3p =
      LoadGame("universal_poker", KuhnLimit3PParameters());
  std::shared_ptr<const Game> os_kuhn_3p =
      LoadGame("kuhn_poker", {{"players", GameParameter(3)}});
  SPIEL_CHECK_GT(kuhn_limit_3p->MaxGameLength(), os_kuhn_3p->MaxGameLength());
  testing::RandomSimTestNoSerialize(*kuhn_limit_3p, 1);
  // TODO(b/145688976): The serialization is also broken
  // In particular, the firstPlayer string "1" is converted back to an integer
  // when deserializing, which crashes.
  // testing::RandomSimTest(*kuhn_limit_3p, 1);
  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker", HoldemNoLimit6PParameters());
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 1);
  testing::RandomSimTest(*holdem_nolimit_6p, 3);
  std::shared_ptr<const Game> holdem_nolimit_fullgame =
      LoadGame(HunlGameString("fullgame"));
  testing::RandomSimTest(*holdem_nolimit_fullgame, 50);
}

void LoadAndRunGameFromGameDef() {
  std::shared_ptr<const Game> holdem_nolimit_6p =
      LoadGame("universal_poker",
               {{"gamedef", GameParameter(std::string(kHoldemNoLimit6P))}});
  testing::RandomSimTestNoSerialize(*holdem_nolimit_6p, 1);
  // TODO(b/145688976): The serialization is also broken
  // testing::RandomSimTest(*holdem_nolimit_6p, 1);
}

void HUNLRegressionTests() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 "
      "50,firstPlayer=2 1 1 "
      "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=400 "
      "400)");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->InformationStateString() << std::endl;
  // Pot bet: call 50, and raise by 200.
  state->ApplyAction(universal_poker::kBet);

  // Now, the minimum bet size is larger than the pot, so player 0 can only
  // fold, call, or go all-in.
  std::vector<Action> actions = state->LegalActions();
  absl::c_sort(actions);

  SPIEL_CHECK_EQ(actions.size(), 3);
  SPIEL_CHECK_EQ(actions[0], universal_poker::kFold);
  SPIEL_CHECK_EQ(actions[1], universal_poker::kCall);
  SPIEL_CHECK_EQ(actions[2], universal_poker::kAllIn);

  // Try a similar test with a stacks of size 300.
  game = LoadGame(
      "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=100 "
      "50,firstPlayer=2 1 1 "
      "1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1,stack=300 "
      "300)");
  state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::cout << state->InformationStateString() << std::endl;

  // The pot bet exactly matches the number of chips available. This is an edge
  // case where all-in is not available, only the pot bet.

  actions = state->LegalActions();
  absl::c_sort(actions);

  SPIEL_CHECK_EQ(actions.size(), 3);
  SPIEL_CHECK_EQ(actions[0], universal_poker::kFold);
  SPIEL_CHECK_EQ(actions[1], universal_poker::kCall);
  SPIEL_CHECK_EQ(actions[2], universal_poker::kBet);
}

void LoadAndRunGameFromDefaultConfig() {
  std::shared_ptr<const Game> game = LoadGame("universal_poker");
  testing::RandomSimTest(*game, 2);
}

void BasicUniversalPokerTests() {
  testing::LoadGameTest("universal_poker");
  testing::ChanceOutcomesTest(*LoadGame("universal_poker"));
  testing::RandomSimTest(*LoadGame("universal_poker"), 100);

  // testing::RandomSimBenchmark("leduc_poker", 10000, false);
  // testing::RandomSimBenchmark("universal_poker", 10000, false);

  testing::CheckChanceOutcomes(*LoadGame("universal_poker"));
}

constexpr absl::string_view kHULHString =
    ("universal_poker(betting=limit,numPlayers=2,numRounds=4,blind=50 100,"
     "firstPlayer=2 1,numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 "
     "1 "
     "1,raiseSize=200 200 400 400,maxRaises=3 4 4 4)");

void ChumpPolicyTests() {
  std::shared_ptr<const Game> game = LoadGame(std::string(kHULHString));
  std::vector<std::unique_ptr<Bot>> bots;
  bots.push_back(MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                               std::make_unique<open_spiel::UniformPolicy>()));
  bots.push_back(
      MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                    std::make_unique<UniformRestrictedActions>(
                        std::vector<ActionType>({ActionType::kCall}))));
  bots.push_back(
      MakePolicyBot(*game, /*player_id=*/0, /*seed=*/0,
                    std::make_unique<UniformRestrictedActions>(
                        std::vector<ActionType>({ActionType::kFold}))));
  bots.push_back(MakePolicyBot(
      *game, /*player_id=*/0, /*seed=*/0,
      std::make_unique<UniformRestrictedActions>(
          std::vector<ActionType>({ActionType::kCall, ActionType::kBet}))));
  for (int i = 0; i < bots.size(); ++i) {
    for (int j = 0; j < bots.size(); ++j) {
      std::unique_ptr<State> state = game->NewInitialState();
      std::vector<Bot *> bots_ptrs = {bots[i].get(), bots[j].get()};
      EvaluateBots(state.get(), bots_ptrs, /*seed=*/42);
    }
  }
}

// Checks min raising functionality.
void FullNLBettingTest1() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
                      "numPlayers=2,"
                      "numRounds=4,"
                      "blind=2 1,"
                      "firstPlayer=2 1 1 1,"
                      "numSuits=4,"
                      "numRanks=13,"
                      "numHoleCards=2,"
                      "numBoardCards=0 3 1 1,"
                      "stack=20 20,"
                      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode())
    state->ApplyAction(state->LegalActions()[0]);  // deal hole cards
  // assert all raise increments are valid
  for (int i = 3; i < 12; ++i)
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 12));
  state->ApplyAction(1);  // call big blind
  state->ApplyAction(1);  // check big blind
  for (int i = 0; i < 3; ++i)
    state->ApplyAction(state->LegalActions()[0]);  // deal flop
  // assert all raise increments are valid
  for (int i = 3; i < 12; ++i)
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  // each player keeps min raising until one is all in
  for (int i = 3; i < 12; ++i)
    state->ApplyAction(i);
  state->ApplyAction(1);  // call last raise
  state->ApplyAction(state->LegalActions()[0]);  // deal turn
  state->ApplyAction(state->LegalActions()[0]);  // deal river
  SPIEL_CHECK_EQ(state->Returns()[0], state->Returns()[1]);  // hand is a draw
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(),
      "ACPC State: STATE:0:cc/r4r6r8r10r12r14r16r18r20c//"
      ":2c2d|2h2s/3c3d3h/3s/4c"));
}

// Checks that raises must double previous bet within the same round but
// each new round resets betting with the min bet size equal to the big blind.
void FullNLBettingTest2() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
                      "numPlayers=2,"
                      "numRounds=4,"
                      "blind=100 50,"
                      "firstPlayer=2 1 1 1,"
                      "numSuits=4,"
                      "numRanks=13,"
                      "numHoleCards=2,"
                      "numBoardCards=0 3 1 1,"
                      "stack=10000 10000,"
                      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);  // deal hole cards
  }
  // assert all raise increments are valid
  for (int i = 3; i < 102; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 102));
  state->ApplyAction(52);  // bet just over half stack
  // raise must double the size of the bet
  // only legal actions now are fold, call, raise all-in
  SPIEL_CHECK_EQ(state->LegalActions().size(), 3);
  state->ApplyAction(1);  // call
  for (int i = 0; i < 3; ++i) {
    state->ApplyAction(state->LegalActions()[0]);  // deal flop
  }
  // new round of betting so we can bet as small as the big blind
  for (int i = 53; i < 102; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  state->ApplyAction(53);  // min bet
  // now we can raise as small as the big blind or as big as an all-in
  for (int i = 54; i < 102; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  state->ApplyAction(1);  // opt just to call
  state->ApplyAction(state->LegalActions()[0]);  // deal turn
  // new round of betting so we can bet as small as the big blind
  for (int i = 55; i < 102; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  state->ApplyAction(55);  // min bet 1 big blind
  state->ApplyAction(57);  // raise to 3 big blinds
  // now a reraise must at least double this raise to 5 big blinds
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 58));
  SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), 59));
  state->ApplyAction(60);  // reraise to 6 big blinds
  // now a reraise must at least double this raise to 9 big blinds
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 62));
  SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), 63));
  state->ApplyAction(1);  // opt to just call
  state->ApplyAction(state->LegalActions()[0]);  // deal river
  // new round of betting so we can bet as small as the big blind
  for (int i = 61; i < 102; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  state->ApplyAction(101);  // all-in!
  state->ApplyAction(0);  // fold
  SPIEL_CHECK_EQ(state->Returns()[0], 5900);
  SPIEL_CHECK_EQ(state->Returns()[1], -5900);
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(),
      "ACPC State: STATE:0:r5100c/r5200c/r5400r5600r5900c/r10000f"
      ":2c2d|2h2s/3c3d3h/3s/4c"));
}

// Checks bet sizing is correct when there are more than two players
// all with different starting stacks.
void FullNLBettingTest3() {
  std::shared_ptr<const Game> game = LoadGame(
      "universal_poker(betting=nolimit,"
                      "numPlayers=3,"
                      "numRounds=4,"
                      "blind=100 50 0,"
                      "firstPlayer=2 1 1 1,"
                      "numSuits=4,"
                      "numRanks=13,"
                      "numHoleCards=2,"
                      "numBoardCards=0 3 1 1,"
                      "stack=500 1000 2000,"
                      "bettingAbstraction=fullgame)");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  state->ApplyAction(1);  // call big blind
  state->ApplyAction(1);  // call big blind
  state->ApplyAction(1);  // check big blind
  for (int i = 0; i < 3; ++i) {
    state->ApplyAction(state->LegalActions()[0]);  // deal flop
  }
  // assert all raise increments are valid
  for (int i = 3; i < 7; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 7));
  state->ApplyAction(1);  // check
  for (int i = 3; i < 12; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 12));
  state->ApplyAction(1);  // check
  for (int i = 3; i < 22; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 22));
  state->ApplyAction(3);  // min raise
  for (int i = 4; i < 7; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 7));
  state->ApplyAction(6);  // short stack goes all-in
  for (int i = 9; i < 12; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 12));
  state->ApplyAction(9);  // min raise
  for (int i = 12; i < 22; ++i) {
    SPIEL_CHECK_TRUE(absl::c_linear_search(state->LegalActions(), i));
  }
  SPIEL_CHECK_FALSE(absl::c_linear_search(state->LegalActions(), 22));
  state->ApplyAction(21);  // all-in
  SPIEL_CHECK_EQ(state->LegalActions().size(), 2);
  state->ApplyAction(1);  // call
  state->ApplyAction(state->LegalActions()[0]);  // deal turn
  state->ApplyAction(state->LegalActions()[0]);  // deal river
  SPIEL_CHECK_EQ(state->Returns()[0], -500);
  SPIEL_CHECK_EQ(state->Returns()[1], -1000);
  SPIEL_CHECK_EQ(state->Returns()[2], 1500);
  SPIEL_CHECK_TRUE(absl::StrContains(state->ToString(),
      "ACPC State: STATE:0:ccc/ccr200r500r800r2000c//"
      ":2c2d|2h2s|3c3d/3h3s4c/4d/4h"));
}

}  // namespace
}  // namespace universal_poker
}  // namespace open_spiel

int main(int argc, char **argv) {
  open_spiel::universal_poker::LoadKuhnLimitWithAndWithoutGameDef();
  open_spiel::universal_poker::LoadHoldemNoLimit6PWithAndWithoutGameDef();
  open_spiel::universal_poker::LoadAndRunGamesFullParameters();
  open_spiel::universal_poker::LoadGameFromDefaultConfig();
  open_spiel::universal_poker::LoadAndRunGameFromGameDef();
  open_spiel::universal_poker::LoadAndRunGameFromDefaultConfig();
  open_spiel::universal_poker::BasicUniversalPokerTests();
  open_spiel::universal_poker::HUNLRegressionTests();
  open_spiel::universal_poker::ChumpPolicyTests();
  open_spiel::universal_poker::FullNLBettingTest1();
  open_spiel::universal_poker::FullNLBettingTest2();
  open_spiel::universal_poker::FullNLBettingTest3();
}
