#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

// категории руки.
// значения совпадают по смыслу с python hand_evaluator.py.
enum class HandCategory : int {
    HIGH_CARD = 1,
    ONE_PAIR = 2,
    TWO_PAIR = 3,
    THREE_OF_A_KIND = 4,
    STRAIGHT = 5,
    FLUSH = 6,
    FULL_HOUSE = 7,
    FOUR_OF_A_KIND = 8,
    STRAIGHT_FLUSH = 9,
};

// компактная карта внутри c++.
// rank: 2..14
// suit: 0..3
struct CardLite {
    int rank;
    int suit;
};

// сравнимое представление силы руки.
struct HandRank {
    HandCategory category;
    std::array<int, 5> tiebreakers;
    int tiebreaker_count;
};

// один weighted combo оппонента в компактном виде.
struct WeightedComboLite {
    int card_a;
    int card_b;
    double weight;
};

// выбранное combo оппонента без веса.
// удобно для hot path.
struct HoleCardsLite {
    int card_a;
    int card_b;
};

// декодируем compact int [0..51] в rank/suit.
CardLite decode_card_id(int card_id) {
    if (card_id < 0 || card_id >= 52) {
        throw std::runtime_error("card id must be in range [0, 51]");
    }

    const int suit = card_id / 13;
    const int rank = (card_id % 13) + 2;

    return CardLite{
        .rank = rank,
        .suit = suit,
    };
}

// бит одной карты в 64-битной маске.
std::uint64_t card_mask_bit(int card_id) {
    if (card_id < 0 || card_id >= 52) {
        throw std::runtime_error("card id must be in range [0, 51]");
    }
    return std::uint64_t{1} << card_id;
}

// сравнение двух рангов рук.
bool hand_rank_greater(const HandRank& left, const HandRank& right) {
    if (static_cast<int>(left.category) != static_cast<int>(right.category)) {
        return static_cast<int>(left.category) > static_cast<int>(right.category);
    }

    const int count = std::max(left.tiebreaker_count, right.tiebreaker_count);
    for (int i = 0; i < count; ++i) {
        const int left_value = i < left.tiebreaker_count ? left.tiebreakers[i] : 0;
        const int right_value = i < right.tiebreaker_count ? right.tiebreakers[i] : 0;

        if (left_value != right_value) {
            return left_value > right_value;
        }
    }

    return false;
}

// python-style compare_hand_ranks: 1 / 0 / -1.
int compare_hand_ranks(const HandRank& left, const HandRank& right) {
    if (hand_rank_greater(left, right)) {
        return 1;
    }
    if (hand_rank_greater(right, left)) {
        return -1;
    }
    return 0;
}

// определяем старшую карту стрита или 0, если стрита нет.
int straight_high_from_counts(const std::array<int, 15>& rank_counts) {
    for (int high = 14; high >= 5; --high) {
        bool ok = true;
        for (int rank = high; rank >= high - 4; --rank) {
            if (rank_counts[rank] == 0) {
                ok = false;
                break;
            }
        }
        if (ok) {
            return high;
        }
    }

    // wheel: A-2-3-4-5
    if (
        rank_counts[14] > 0 &&
        rank_counts[5] > 0 &&
        rank_counts[4] > 0 &&
        rank_counts[3] > 0 &&
        rank_counts[2] > 0
    ) {
        return 5;
    }

    return 0;
}

// собрать top-n rank'ов из rank_counts.
// rank_counts[rank] может быть больше 1, но здесь нам нужны именно distinct ranks.
std::array<int, 5> top_distinct_ranks_from_counts(
    const std::array<int, 15>& rank_counts,
    int needed
) {
    std::array<int, 5> result{};
    int filled = 0;

    for (int rank = 14; rank >= 2 && filled < needed; --rank) {
        if (rank_counts[rank] > 0) {
            result[filled] = rank;
            filled += 1;
        }
    }

    return result;
}

// собрать top-n rank'ов конкретной масти.
std::array<int, 5> top_flush_ranks_from_suit_counts(
    const std::array<int, 15>& suit_rank_counts,
    int needed
) {
    std::array<int, 5> result{};
    int filled = 0;

    for (int rank = 14; rank >= 2 && filled < needed; --rank) {
        if (suit_rank_counts[rank] > 0) {
            result[filled] = rank;
            filled += 1;
        }
    }

    return result;
}

// оценка 7 карт через перебор 21 пятёрки.
HandRank evaluate_best_of_seven(const std::array<CardLite, 7>& cards) {
    std::array<int, 15> rank_counts{};
    std::array<int, 4> suit_counts{};
    std::array<std::array<int, 15>, 4> suit_rank_counts{};

    for (const auto& card : cards) {
        rank_counts[card.rank] += 1;
        suit_counts[card.suit] += 1;
        suit_rank_counts[card.suit][card.rank] += 1;
    }

    // 1. straight flush
    for (int suit = 0; suit < 4; ++suit) {
        if (suit_counts[suit] >= 5) {
            const int straight_flush_high = straight_high_from_counts(suit_rank_counts[suit]);
            if (straight_flush_high > 0) {
                return HandRank{
                    .category = HandCategory::STRAIGHT_FLUSH,
                    .tiebreakers = std::array<int, 5>{straight_flush_high, 0, 0, 0, 0},
                    .tiebreaker_count = 1,
                };
            }
        }
    }

    int top_quad = 0;
    std::array<int, 2> trips{};
    int trips_count = 0;
    std::array<int, 3> pairs{};
    int pairs_count = 0;

    for (int rank = 14; rank >= 2; --rank) {
        const int count = rank_counts[rank];

        if (count == 4) {
            if (top_quad == 0) {
                top_quad = rank;
            }
        } else if (count == 3) {
            if (trips_count < 2) {
                trips[trips_count] = rank;
                trips_count += 1;
            }
        } else if (count == 2) {
            if (pairs_count < 3) {
                pairs[pairs_count] = rank;
                pairs_count += 1;
            }
        }
    }

    // 2. four of a kind
    if (top_quad > 0) {
        int kicker = 0;
        for (int rank = 14; rank >= 2; --rank) {
            if (rank != top_quad && rank_counts[rank] > 0) {
                kicker = rank;
                break;
            }
        }

        return HandRank{
            .category = HandCategory::FOUR_OF_A_KIND,
            .tiebreakers = std::array<int, 5>{top_quad, kicker, 0, 0, 0},
            .tiebreaker_count = 2,
        };
    }

    // 3. full house
    if (trips_count >= 1 && (trips_count >= 2 || pairs_count >= 1)) {
        const int top_trips = trips[0];
        const int top_pair = trips_count >= 2 ? trips[1] : pairs[0];

        return HandRank{
            .category = HandCategory::FULL_HOUSE,
            .tiebreakers = std::array<int, 5>{top_trips, top_pair, 0, 0, 0},
            .tiebreaker_count = 2,
        };
    }

    // 4. flush
    for (int suit = 0; suit < 4; ++suit) {
        if (suit_counts[suit] >= 5) {
            const std::array<int, 5> flush_ranks =
                top_flush_ranks_from_suit_counts(suit_rank_counts[suit], 5);

            return HandRank{
                .category = HandCategory::FLUSH,
                .tiebreakers = flush_ranks,
                .tiebreaker_count = 5,
            };
        }
    }

    // 5. straight
    const int straight_high = straight_high_from_counts(rank_counts);
    if (straight_high > 0) {
        return HandRank{
            .category = HandCategory::STRAIGHT,
            .tiebreakers = std::array<int, 5>{straight_high, 0, 0, 0, 0},
            .tiebreaker_count = 1,
        };
    }

    // 6. three of a kind
    if (trips_count >= 1) {
        const int top_trips = trips[0];

        std::array<int, 2> kickers{};
        int kicker_count = 0;

        for (int rank = 14; rank >= 2 && kicker_count < 2; --rank) {
            if (rank != top_trips && rank_counts[rank] > 0) {
                kickers[kicker_count] = rank;
                kicker_count += 1;
            }
        }

        return HandRank{
            .category = HandCategory::THREE_OF_A_KIND,
            .tiebreakers = std::array<int, 5>{top_trips, kickers[0], kickers[1], 0, 0},
            .tiebreaker_count = 3,
        };
    }

    // 7. two pair
    if (pairs_count >= 2) {
        const int high_pair = pairs[0];
        const int low_pair = pairs[1];

        int kicker = 0;
        for (int rank = 14; rank >= 2; --rank) {
            if (rank != high_pair && rank != low_pair && rank_counts[rank] > 0) {
                kicker = rank;
                break;
            }
        }

        return HandRank{
            .category = HandCategory::TWO_PAIR,
            .tiebreakers = std::array<int, 5>{high_pair, low_pair, kicker, 0, 0},
            .tiebreaker_count = 3,
        };
    }

    // 8. one pair
    if (pairs_count >= 1) {
        const int pair_rank = pairs[0];

        std::array<int, 3> kickers{};
        int kicker_count = 0;

        for (int rank = 14; rank >= 2 && kicker_count < 3; --rank) {
            if (rank != pair_rank && rank_counts[rank] > 0) {
                kickers[kicker_count] = rank;
                kicker_count += 1;
            }
        }

        return HandRank{
            .category = HandCategory::ONE_PAIR,
            .tiebreakers = std::array<int, 5>{pair_rank, kickers[0], kickers[1], kickers[2], 0},
            .tiebreaker_count = 4,
        };
    }

    // 9. high card
    const std::array<int, 5> high_cards = top_distinct_ranks_from_counts(rank_counts, 5);
    return HandRank{
        .category = HandCategory::HIGH_CARD,
        .tiebreakers = high_cards,
        .tiebreaker_count = 5,
    };
}

// helper: собираем 7 карт игрока из 2 hole и 5 board.
std::array<CardLite, 7> build_seven_cards(
    const std::array<int, 2>& hole_cards,
    const std::array<int, 5>& board_cards
) {
    return std::array<CardLite, 7>{
        decode_card_id(hole_cards[0]),
        decode_card_id(hole_cards[1]),
        decode_card_id(board_cards[0]),
        decode_card_id(board_cards[1]),
        decode_card_id(board_cards[2]),
        decode_card_id(board_cards[3]),
        decode_card_id(board_cards[4]),
    };
}

// более дешёвый helper для hot path.
// не создаёт новых vector, а сразу собирает std::array из 7 карт.
HandRank evaluate_seven_from_ids(
    int hole_a,
    int hole_b,
    const std::array<int, 5>& board
) {
    const std::array<CardLite, 7> cards{
        decode_card_id(hole_a),
        decode_card_id(hole_b),
        decode_card_id(board[0]),
        decode_card_id(board[1]),
        decode_card_id(board[2]),
        decode_card_id(board[3]),
        decode_card_id(board[4]),
    };

    return evaluate_best_of_seven(cards);
}

// читаем tuple из двух card ids.
std::array<int, 2> parse_two_card_tuple(const py::handle& obj) {
    py::tuple cards = py::reinterpret_borrow<py::tuple>(obj);
    if (py::len(cards) != 2) {
        throw std::runtime_error("two-card tuple must contain exactly 2 items");
    }

    return std::array<int, 2>{
        cards[0].cast<int>(),
        cards[1].cast<int>(),
    };
}

// читаем board из 5 карт.
std::array<int, 5> parse_five_card_tuple(const py::tuple& obj) {
    if (py::len(obj) != 5) {
        throw std::runtime_error("five-card tuple must contain exactly 5 items");
    }

    return std::array<int, 5>{
        obj[0].cast<int>(),
        obj[1].cast<int>(),
        obj[2].cast<int>(),
        obj[3].cast<int>(),
        obj[4].cast<int>(),
    };
}

// читаем board из 4 карт для turn.
std::array<int, 4> parse_four_card_tuple(const py::tuple& obj) {
    if (py::len(obj) != 4) {
        throw std::runtime_error("four-card tuple must contain exactly 4 items");
    }

    return std::array<int, 4>{
        obj[0].cast<int>(),
        obj[1].cast<int>(),
        obj[2].cast<int>(),
        obj[3].cast<int>(),
    };
}

// читаем board из 3 карт для flop.
std::array<int, 3> parse_three_card_tuple(const py::tuple& obj) {
    if (py::len(obj) != 3) {
        throw std::runtime_error("three-card tuple must contain exactly 3 items");
    }

    return std::array<int, 3>{
        obj[0].cast<int>(),
        obj[1].cast<int>(),
        obj[2].cast<int>(),
    };
}

// helper: собираем полный board turn + river.
std::array<int, 5> make_full_board_from_turn(
    const std::array<int, 4>& turn_board,
    int river_card
) {
    return std::array<int, 5>{
        turn_board[0],
        turn_board[1],
        turn_board[2],
        turn_board[3],
        river_card,
    };
}

// helper: собираем полный board flop + turn + river.
std::array<int, 5> make_full_board_from_flop(
    const std::array<int, 3>& flop_board,
    int turn_card,
    int river_card
) {
    return std::array<int, 5>{
        flop_board[0],
        flop_board[1],
        flop_board[2],
        turn_card,
        river_card,
    };
}

// читаем один weighted combo item.
WeightedComboLite parse_weighted_combo_item(const py::handle& obj) {
    py::tuple item = py::reinterpret_borrow<py::tuple>(obj);
    if (py::len(item) != 3) {
        throw std::runtime_error("weighted combo item must contain 3 items");
    }

    const int card_a = item[0].cast<int>();
    const int card_b = item[1].cast<int>();
    const double weight = item[2].cast<double>();

    if (card_a == card_b) {
        throw std::runtime_error("weighted combo item cannot contain the same card twice");
    }

    if (weight < 0.0) {
        throw std::runtime_error("weighted combo item must have non-negative weight");
    }

    return WeightedComboLite{
        .card_a = card_a,
        .card_b = card_b,
        .weight = weight,
    };
}

// читаем весь диапазон одного оппонента.
std::vector<WeightedComboLite> parse_weighted_range(const py::handle& obj) {
    py::tuple range = py::reinterpret_borrow<py::tuple>(obj);
    std::vector<WeightedComboLite> result;
    result.reserve(py::len(range));

    for (const py::handle& item_handle : range) {
        WeightedComboLite combo = parse_weighted_combo_item(item_handle);
        if (combo.weight > 0.0) {
            result.push_back(combo);
        }
    }

    return result;
}

// showdown героя против всех villains на полном board.
// использует fixed-size массив выбранных hole cards,
// чтобы не создавать vector<HandRank> на каждом sample.
int compare_hero_vs_selected_villains(
    const HandRank& hero_rank,
    const std::array<HoleCardsLite, 4>& selected_villains,
    int villain_count,
    const std::array<int, 5>& board
) {
    bool has_tie = false;

    for (int i = 0; i < villain_count; ++i) {
        const HandRank villain_rank = evaluate_seven_from_ids(
            selected_villains[i].card_a,
            selected_villains[i].card_b,
            board
        );

        const int comparison = compare_hand_ranks(hero_rank, villain_rank);

        if (comparison < 0) {
            return -1;
        }
        if (comparison == 0) {
            has_tie = true;
        }
    }

    if (has_tie) {
        return 0;
    }

    return 1;
}

// showdown героя против всех villains на полном board.
// старый helper нужен exact path.
int compare_hero_vs_villains(
    const HandRank& hero_rank,
    const std::vector<HandRank>& villain_ranks
) {
    bool has_tie = false;

    for (const HandRank& villain_rank : villain_ranks) {
        const int comparison = compare_hand_ranks(hero_rank, villain_rank);

        if (comparison < 0) {
            return -1;
        }
        if (comparison == 0) {
            has_tie = true;
        }
    }

    if (has_tie) {
        return 0;
    }

    return 1;
}

// weighted choice по доступным индексам.
// это простой cumulative-sum sampler.
std::size_t sample_weighted_index(
    const std::vector<WeightedComboLite>& combos,
    const std::vector<std::size_t>& eligible_indexes,
    std::mt19937_64& rng
) {
    double total_weight = 0.0;
    for (std::size_t index : eligible_indexes) {
        total_weight += combos[index].weight;
    }

    if (total_weight <= 0.0) {
        throw std::runtime_error("sample_weighted_index requires positive total weight");
    }

    std::uniform_real_distribution<double> dist(0.0, total_weight);
    const double threshold = dist(rng);

    double cumulative = 0.0;
    for (std::size_t index : eligible_indexes) {
        cumulative += combos[index].weight;
        if (cumulative >= threshold) {
            return index;
        }
    }

    return eligible_indexes.back();
}

// sampling совместимых villain combos.
// теперь используем fixed-size массив,
// чтобы не делать лишние vector allocations в MC hot loop.
bool sample_compatible_villain_holes_fixed(
    const std::vector<std::vector<WeightedComboLite>>& villain_ranges,
    std::uint64_t initial_used_mask,
    std::array<HoleCardsLite, 4>& selected_villains,
    int& selected_count,
    std::mt19937_64& rng
) {
    selected_count = 0;
    std::uint64_t used_mask = initial_used_mask;

    for (const auto& villain_range : villain_ranges) {
        std::vector<std::size_t> eligible_indexes;
        eligible_indexes.reserve(villain_range.size());

        for (std::size_t i = 0; i < villain_range.size(); ++i) {
            const auto& combo = villain_range[i];
            const std::uint64_t combo_mask =
                card_mask_bit(combo.card_a) | card_mask_bit(combo.card_b);

            if ((combo_mask & used_mask) != 0) {
                continue;
            }

            eligible_indexes.push_back(i);
        }

        if (eligible_indexes.empty()) {
            return false;
        }

        const std::size_t chosen_index = sample_weighted_index(villain_range, eligible_indexes, rng);
        const auto& chosen_combo = villain_range[chosen_index];

        selected_villains[selected_count] = HoleCardsLite{
            .card_a = chosen_combo.card_a,
            .card_b = chosen_combo.card_b,
        };
        selected_count += 1;

        used_mask |= card_mask_bit(chosen_combo.card_a);
        used_mask |= card_mask_bit(chosen_combo.card_b);
    }

    return true;
}

// выбрать одну карту из доступных.
// работает через размер пула и random index.
int sample_one_card_from_available_fixed(
    const std::array<int, 52>& available_cards,
    int available_count,
    std::mt19937_64& rng
) {
    if (available_count <= 0) {
        throw std::runtime_error("sample_one_card_from_available_fixed requires non-empty pool");
    }

    std::uniform_int_distribution<int> dist(0, available_count - 1);
    return available_cards[dist(rng)];
}

// выбрать две разные карты из доступных без повторов.
std::array<int, 2> sample_two_cards_from_available_fixed(
    const std::array<int, 52>& available_cards,
    int available_count,
    std::mt19937_64& rng
) {
    if (available_count < 2) {
        throw std::runtime_error("sample_two_cards_from_available_fixed requires at least 2 cards");
    }

    std::uniform_int_distribution<int> first_dist(0, available_count - 1);
    const int first_index = first_dist(rng);
    const int first_card = available_cards[first_index];

    std::uniform_int_distribution<int> second_dist(0, available_count - 2);
    int second_index = second_dist(rng);

    // пропускаем already chosen index без копирования массива.
    if (second_index >= first_index) {
        second_index += 1;
    }

    const int second_card = available_cards[second_index];

    return std::array<int, 2>{first_card, second_card};
}

// построить список доступных карт по used_mask в fixed-size массив.
int build_available_cards_fixed(
    std::uint64_t used_mask,
    std::array<int, 52>& available_cards
) {
    int count = 0;

    for (int card_id = 0; card_id < 52; ++card_id) {
        if ((card_mask_bit(card_id) & used_mask) == 0) {
            available_cards[count] = card_id;
            count += 1;
        }
    }

    return count;
}

// рекурсивный перебор совместимых villain combo selection для river.
void accumulate_multiway_river_exact(
    const std::vector<std::vector<WeightedComboLite>>& villain_ranges,
    std::size_t villain_index,
    std::uint64_t used_mask,
    double selection_weight,
    const std::array<int, 5>& board,
    const HandRank& hero_rank,
    std::vector<HandRank>& selected_villain_ranks,
    double& total_weighted_wins,
    double& total_weighted_ties,
    double& total_weighted_losses,
    double& total_weight
) {
    if (villain_index == villain_ranges.size()) {
        const int showdown = compare_hero_vs_villains(hero_rank, selected_villain_ranks);

        if (showdown > 0) {
            total_weighted_wins += selection_weight;
        } else if (showdown < 0) {
            total_weighted_losses += selection_weight;
        } else {
            total_weighted_ties += selection_weight;
        }

        total_weight += selection_weight;
        return;
    }

    const auto& villain_range = villain_ranges[villain_index];

    for (const WeightedComboLite& combo : villain_range) {
        const std::uint64_t combo_mask =
            card_mask_bit(combo.card_a) | card_mask_bit(combo.card_b);

        if ((combo_mask & used_mask) != 0) {
            continue;
        }

        const std::array<int, 2> villain_hole{combo.card_a, combo.card_b};
        const std::array<CardLite, 7> villain_cards = build_seven_cards(villain_hole, board);
        const HandRank villain_rank = evaluate_best_of_seven(villain_cards);

        selected_villain_ranks.push_back(villain_rank);

        accumulate_multiway_river_exact(
            villain_ranges,
            villain_index + 1,
            used_mask | combo_mask,
            selection_weight * combo.weight,
            board,
            hero_rank,
            selected_villain_ranks,
            total_weighted_wins,
            total_weighted_ties,
            total_weighted_losses,
            total_weight
        );

        selected_villain_ranks.pop_back();
    }
}

// turn exact.
void accumulate_multiway_turn_exact(
    const std::vector<std::vector<WeightedComboLite>>& villain_ranges,
    std::size_t villain_index,
    std::uint64_t used_mask,
    double selection_weight,
    const std::array<int, 2>& hero_hole,
    const std::array<int, 4>& turn_board,
    std::vector<std::array<int, 2>>& selected_villain_holes,
    double& total_weighted_wins,
    double& total_weighted_ties,
    double& total_weighted_losses,
    double& total_weight
) {
    if (villain_index == villain_ranges.size()) {
        for (int river_card = 0; river_card < 52; ++river_card) {
            const std::uint64_t river_mask = card_mask_bit(river_card);

            if ((river_mask & used_mask) != 0) {
                continue;
            }

            const std::array<int, 5> full_board = make_full_board_from_turn(turn_board, river_card);

            const std::array<CardLite, 7> hero_cards = build_seven_cards(hero_hole, full_board);
            const HandRank hero_rank = evaluate_best_of_seven(hero_cards);

            std::vector<HandRank> villain_ranks;
            villain_ranks.reserve(selected_villain_holes.size());

            for (const auto& villain_hole : selected_villain_holes) {
                const std::array<CardLite, 7> villain_cards =
                    build_seven_cards(villain_hole, full_board);
                villain_ranks.push_back(evaluate_best_of_seven(villain_cards));
            }

            const int showdown = compare_hero_vs_villains(hero_rank, villain_ranks);

            if (showdown > 0) {
                total_weighted_wins += selection_weight;
            } else if (showdown < 0) {
                total_weighted_losses += selection_weight;
            } else {
                total_weighted_ties += selection_weight;
            }

            total_weight += selection_weight;
        }

        return;
    }

    const auto& villain_range = villain_ranges[villain_index];

    for (const WeightedComboLite& combo : villain_range) {
        const std::uint64_t combo_mask =
            card_mask_bit(combo.card_a) | card_mask_bit(combo.card_b);

        if ((combo_mask & used_mask) != 0) {
            continue;
        }

        selected_villain_holes.push_back(std::array<int, 2>{combo.card_a, combo.card_b});

        accumulate_multiway_turn_exact(
            villain_ranges,
            villain_index + 1,
            used_mask | combo_mask,
            selection_weight * combo.weight,
            hero_hole,
            turn_board,
            selected_villain_holes,
            total_weighted_wins,
            total_weighted_ties,
            total_weighted_losses,
            total_weight
        );

        selected_villain_holes.pop_back();
    }
}

// exact heads-up flop.
py::dict calculate_exact_weighted_postflop_equity_flop_heads_up(
    const py::tuple& hero_card_ids,
    const py::tuple& board_card_ids,
    const py::tuple& villain_ranges_obj
) {
    if (py::len(villain_ranges_obj) != 1) {
        throw std::runtime_error("flop heads-up exact path expects exactly 1 villain range");
    }

    const std::array<int, 2> hero_hole = parse_two_card_tuple(hero_card_ids);
    const std::array<int, 3> flop_board = parse_three_card_tuple(board_card_ids);

    const std::vector<WeightedComboLite> villain_range =
        parse_weighted_range(villain_ranges_obj[0]);

    const std::uint64_t board_mask =
        card_mask_bit(flop_board[0]) |
        card_mask_bit(flop_board[1]) |
        card_mask_bit(flop_board[2]);

    const std::uint64_t hero_mask =
        card_mask_bit(hero_hole[0]) |
        card_mask_bit(hero_hole[1]);

    if ((hero_mask & board_mask) != 0) {
        throw std::runtime_error("hero hole cards must not conflict with board cards");
    }

    double total_weighted_wins = 0.0;
    double total_weighted_ties = 0.0;
    double total_weighted_losses = 0.0;
    double total_weight = 0.0;

    for (const WeightedComboLite& combo : villain_range) {
        const std::uint64_t combo_mask =
            card_mask_bit(combo.card_a) | card_mask_bit(combo.card_b);

        if ((combo_mask & (hero_mask | board_mask)) != 0) {
            continue;
        }

        const std::uint64_t used_mask = hero_mask | board_mask | combo_mask;
        const std::array<int, 2> villain_hole{combo.card_a, combo.card_b};

        for (int turn_card = 0; turn_card < 52; ++turn_card) {
            const std::uint64_t turn_mask = card_mask_bit(turn_card);

            if ((turn_mask & used_mask) != 0) {
                continue;
            }

            for (int river_card = turn_card + 1; river_card < 52; ++river_card) {
                const std::uint64_t river_mask = card_mask_bit(river_card);

                if ((river_mask & used_mask) != 0) {
                    continue;
                }

                const std::array<int, 5> full_board =
                    make_full_board_from_flop(flop_board, turn_card, river_card);

                const HandRank hero_rank = evaluate_seven_from_ids(
                    hero_hole[0],
                    hero_hole[1],
                    full_board
                );
                const HandRank villain_rank = evaluate_seven_from_ids(
                    villain_hole[0],
                    villain_hole[1],
                    full_board
                );

                const int comparison = compare_hand_ranks(hero_rank, villain_rank);

                if (comparison > 0) {
                    total_weighted_wins += combo.weight;
                } else if (comparison < 0) {
                    total_weighted_losses += combo.weight;
                } else {
                    total_weighted_ties += combo.weight;
                }

                total_weight += combo.weight;
            }
        }
    }

    if (total_weight <= 0.0) {
        throw std::runtime_error(
            "No valid scenarios were produced in native flop heads-up exact path"
        );
    }

    const double win_rate = total_weighted_wins / total_weight;
    const double tie_rate = total_weighted_ties / total_weight;
    const double loss_rate = total_weighted_losses / total_weight;
    const double hero_equity = win_rate + 0.5 * tie_rate;

    py::dict result;
    result["hero_equity"] = hero_equity;
    result["win_rate"] = win_rate;
    result["tie_rate"] = tie_rate;
    result["loss_rate"] = loss_rate;
    result["scenarios_evaluated"] = static_cast<std::int64_t>(total_weight + 0.5);
    return result;
}

// river exact для 1..4 villains.
py::dict calculate_exact_weighted_postflop_equity_river(
    const py::tuple& hero_card_ids,
    const py::tuple& board_card_ids,
    const py::tuple& villain_ranges_obj
) {
    const std::array<int, 2> hero_hole = parse_two_card_tuple(hero_card_ids);
    const std::array<int, 5> board = parse_five_card_tuple(board_card_ids);

    std::vector<std::vector<WeightedComboLite>> villain_ranges;
    villain_ranges.reserve(py::len(villain_ranges_obj));

    for (const py::handle& range_handle : villain_ranges_obj) {
        villain_ranges.push_back(parse_weighted_range(range_handle));
    }

    const std::uint64_t board_mask =
        card_mask_bit(board[0]) |
        card_mask_bit(board[1]) |
        card_mask_bit(board[2]) |
        card_mask_bit(board[3]) |
        card_mask_bit(board[4]);

    const std::uint64_t hero_mask =
        card_mask_bit(hero_hole[0]) |
        card_mask_bit(hero_hole[1]);

    if ((hero_mask & board_mask) != 0) {
        throw std::runtime_error("hero hole cards must not conflict with board cards");
    }

    const std::array<CardLite, 7> hero_cards = build_seven_cards(hero_hole, board);
    const HandRank hero_rank = evaluate_best_of_seven(hero_cards);

    double total_weighted_wins = 0.0;
    double total_weighted_ties = 0.0;
    double total_weighted_losses = 0.0;
    double total_weight = 0.0;

    std::vector<HandRank> selected_villain_ranks;
    selected_villain_ranks.reserve(villain_ranges.size());

    const std::uint64_t used_mask = hero_mask | board_mask;

    accumulate_multiway_river_exact(
        villain_ranges,
        0,
        used_mask,
        1.0,
        board,
        hero_rank,
        selected_villain_ranks,
        total_weighted_wins,
        total_weighted_ties,
        total_weighted_losses,
        total_weight
    );

    if (total_weight <= 0.0) {
        throw std::runtime_error("No valid scenarios were produced in native river exact path");
    }

    const double win_rate = total_weighted_wins / total_weight;
    const double tie_rate = total_weighted_ties / total_weight;
    const double loss_rate = total_weighted_losses / total_weight;
    const double hero_equity = win_rate + 0.5 * tie_rate;

    py::dict result;
    result["hero_equity"] = hero_equity;
    result["win_rate"] = win_rate;
    result["tie_rate"] = tie_rate;
    result["loss_rate"] = loss_rate;
    result["scenarios_evaluated"] = static_cast<std::int64_t>(total_weight + 0.5);
    return result;
}

// turn exact для 1..4 villains.
py::dict calculate_exact_weighted_postflop_equity_turn(
    const py::tuple& hero_card_ids,
    const py::tuple& board_card_ids,
    const py::tuple& villain_ranges_obj
) {
    const std::array<int, 2> hero_hole = parse_two_card_tuple(hero_card_ids);
    const std::array<int, 4> turn_board = parse_four_card_tuple(board_card_ids);

    std::vector<std::vector<WeightedComboLite>> villain_ranges;
    villain_ranges.reserve(py::len(villain_ranges_obj));

    for (const py::handle& range_handle : villain_ranges_obj) {
        villain_ranges.push_back(parse_weighted_range(range_handle));
    }

    const std::uint64_t board_mask =
        card_mask_bit(turn_board[0]) |
        card_mask_bit(turn_board[1]) |
        card_mask_bit(turn_board[2]) |
        card_mask_bit(turn_board[3]);

    const std::uint64_t hero_mask =
        card_mask_bit(hero_hole[0]) |
        card_mask_bit(hero_hole[1]);

    if ((hero_mask & board_mask) != 0) {
        throw std::runtime_error("hero hole cards must not conflict with board cards");
    }

    double total_weighted_wins = 0.0;
    double total_weighted_ties = 0.0;
    double total_weighted_losses = 0.0;
    double total_weight = 0.0;

    std::vector<std::array<int, 2>> selected_villain_holes;
    selected_villain_holes.reserve(villain_ranges.size());

    const std::uint64_t used_mask = hero_mask | board_mask;

    accumulate_multiway_turn_exact(
        villain_ranges,
        0,
        used_mask,
        1.0,
        hero_hole,
        turn_board,
        selected_villain_holes,
        total_weighted_wins,
        total_weighted_ties,
        total_weighted_losses,
        total_weight
    );

    if (total_weight <= 0.0) {
        throw std::runtime_error("No valid scenarios were produced in native turn exact path");
    }

    const double win_rate = total_weighted_wins / total_weight;
    const double tie_rate = total_weighted_ties / total_weight;
    const double loss_rate = total_weighted_losses / total_weight;
    const double hero_equity = win_rate + 0.5 * tie_rate;

    py::dict result;
    result["hero_equity"] = hero_equity;
    result["win_rate"] = win_rate;
    result["tie_rate"] = tie_rate;
    result["loss_rate"] = loss_rate;
    result["scenarios_evaluated"] = static_cast<std::int64_t>(total_weight + 0.5);
    return result;
}

// native monte carlo support for flop/turn/river.
bool supports_monte_carlo_weighted_postflop(
    int villain_count,
    int board_size
) {
    return villain_count >= 1 && villain_count <= 4 && board_size >= 3 && board_size <= 5;
}

// native monte carlo.
// оптимизации здесь:
// - fixed-size selected_villains вместо vector
// - fixed-size available_cards вместо vector<int>
// - fixed-size showdown compare без vector<HandRank>
// - evaluate_seven_from_ids вместо части промежуточных helpers
py::dict calculate_monte_carlo_weighted_postflop_equity(
    py::tuple hero_card_ids,
    py::tuple board_card_ids,
    py::tuple villain_ranges_obj,
    int sample_count,
    py::object random_seed_obj
) {
    if (sample_count <= 0) {
        throw std::runtime_error("sample_count must be positive");
    }

    const std::array<int, 2> hero_hole = parse_two_card_tuple(hero_card_ids);

    std::vector<int> board_cards;
    board_cards.reserve(py::len(board_card_ids));
    for (const py::handle& card_handle : board_card_ids) {
        board_cards.push_back(card_handle.cast<int>());
    }

    if (board_cards.size() < 3 || board_cards.size() > 5) {
        throw std::runtime_error("board_card_ids must contain from 3 to 5 items");
    }

    std::vector<std::vector<WeightedComboLite>> villain_ranges;
    villain_ranges.reserve(py::len(villain_ranges_obj));

    for (const py::handle& range_handle : villain_ranges_obj) {
        villain_ranges.push_back(parse_weighted_range(range_handle));
    }

    std::uint64_t board_mask = 0;
    for (int board_card : board_cards) {
        board_mask |= card_mask_bit(board_card);
    }

    const std::uint64_t hero_mask =
        card_mask_bit(hero_hole[0]) |
        card_mask_bit(hero_hole[1]);

    if ((hero_mask & board_mask) != 0) {
        throw std::runtime_error("hero hole cards must not conflict with board cards");
    }

    const std::uint64_t initial_used_mask = hero_mask | board_mask;

    std::uint64_t seed_value = 42;
    if (!random_seed_obj.is_none()) {
        seed_value = static_cast<std::uint64_t>(random_seed_obj.cast<long long>());
    }
    std::mt19937_64 rng(seed_value);

    int wins = 0;
    int ties = 0;
    int losses = 0;
    int successful_samples = 0;

    const int max_attempts = std::max(sample_count * 20, 1000);
    int attempts = 0;

    std::array<HoleCardsLite, 4> selected_villains{};
    int selected_count = 0;

    std::array<int, 52> available_cards{};

    while (successful_samples < sample_count && attempts < max_attempts) {
        attempts += 1;

        const bool sampled = sample_compatible_villain_holes_fixed(
            villain_ranges,
            initial_used_mask,
            selected_villains,
            selected_count,
            rng
        );
        if (!sampled) {
            continue;
        }

        std::uint64_t used_mask = initial_used_mask;
        for (int i = 0; i < selected_count; ++i) {
            used_mask |= card_mask_bit(selected_villains[i].card_a);
            used_mask |= card_mask_bit(selected_villains[i].card_b);
        }

        const int available_count = build_available_cards_fixed(used_mask, available_cards);

        std::array<int, 5> full_board{};

        if (board_cards.size() == 5) {
            full_board = std::array<int, 5>{
                board_cards[0],
                board_cards[1],
                board_cards[2],
                board_cards[3],
                board_cards[4],
            };
        } else if (board_cards.size() == 4) {
            const int river_card = sample_one_card_from_available_fixed(
                available_cards,
                available_count,
                rng
            );

            full_board = std::array<int, 5>{
                board_cards[0],
                board_cards[1],
                board_cards[2],
                board_cards[3],
                river_card,
            };
        } else {
            const std::array<int, 2> sampled_cards =
                sample_two_cards_from_available_fixed(
                    available_cards,
                    available_count,
                    rng
                );

            full_board = std::array<int, 5>{
                board_cards[0],
                board_cards[1],
                board_cards[2],
                sampled_cards[0],
                sampled_cards[1],
            };
        }

        const HandRank hero_rank = evaluate_seven_from_ids(
            hero_hole[0],
            hero_hole[1],
            full_board
        );

        const int showdown = compare_hero_vs_selected_villains(
            hero_rank,
            selected_villains,
            selected_count,
            full_board
        );

        if (showdown > 0) {
            wins += 1;
        } else if (showdown < 0) {
            losses += 1;
        } else {
            ties += 1;
        }

        successful_samples += 1;
    }

    if (successful_samples <= 0) {
        throw std::runtime_error("Native monte carlo calculation produced no valid samples");
    }

    const double win_rate = static_cast<double>(wins) / static_cast<double>(successful_samples);
    const double tie_rate = static_cast<double>(ties) / static_cast<double>(successful_samples);
    const double loss_rate = static_cast<double>(losses) / static_cast<double>(successful_samples);
    const double hero_equity = win_rate + 0.5 * tie_rate;

    py::dict result;
    result["hero_equity"] = hero_equity;
    result["win_rate"] = win_rate;
    result["tie_rate"] = tie_rate;
    result["loss_rate"] = loss_rate;
    result["scenarios_evaluated"] = successful_samples;
    return result;
}

}  // namespace

// exact support:
// - flop only heads-up
// - turn 1..4 villains
// - river 1..4 villains
static bool supports_exact_weighted_postflop(
    int villain_count,
    int board_size
) {
    if (board_size == 3) {
        return villain_count == 1;
    }

    return villain_count >= 1 && villain_count <= 4 && (board_size == 4 || board_size == 5);
}

static py::dict calculate_exact_weighted_postflop_equity(
    py::tuple hero_card_ids,
    py::tuple board_card_ids,
    py::tuple villain_ranges
) {
    if (py::len(hero_card_ids) != 2) {
        throw std::runtime_error("hero_card_ids must contain exactly 2 items");
    }

    if (py::len(board_card_ids) < 3 || py::len(board_card_ids) > 5) {
        throw std::runtime_error("board_card_ids must contain from 3 to 5 items");
    }

    if (py::len(villain_ranges) < 1 || py::len(villain_ranges) > 4) {
        throw std::runtime_error("villain_ranges must contain from 1 to 4 ranges");
    }

    if (py::len(board_card_ids) == 3 && py::len(villain_ranges) == 1) {
        return calculate_exact_weighted_postflop_equity_flop_heads_up(
            hero_card_ids,
            board_card_ids,
            villain_ranges
        );
    }

    if (py::len(board_card_ids) == 5) {
        return calculate_exact_weighted_postflop_equity_river(
            hero_card_ids,
            board_card_ids,
            villain_ranges
        );
    }

    if (py::len(board_card_ids) == 4) {
        return calculate_exact_weighted_postflop_equity_turn(
            hero_card_ids,
            board_card_ids,
            villain_ranges
        );
    }

    throw std::runtime_error(
        "calculate_exact_weighted_postflop_equity does not support this spot yet"
    );
}

PYBIND11_MODULE(_native_equity, m) {
    m.doc() = "Native equity backend module";

    m.def(
        "supports_exact_weighted_postflop",
        &supports_exact_weighted_postflop,
        "Check whether native backend supports exact weighted postflop spot"
    );

    m.def(
        "calculate_exact_weighted_postflop_equity",
        &calculate_exact_weighted_postflop_equity,
        "Calculate exact weighted postflop equity"
    );

    m.def(
        "supports_monte_carlo_weighted_postflop",
        &supports_monte_carlo_weighted_postflop,
        "Check whether native backend supports monte carlo weighted postflop spot"
    );

    m.def(
        "calculate_monte_carlo_weighted_postflop_equity",
        &calculate_monte_carlo_weighted_postflop_equity,
        "Calculate monte carlo weighted postflop equity"
    );
}
