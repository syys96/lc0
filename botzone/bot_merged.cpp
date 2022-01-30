// ######## begin of system include headers ########


#include <string_view>
#include <fstream>
#include <zlib.h>
#include <mutex>
#include <cmath>
#include <intrin.h>
#include <functional>
#include <condition_variable>
#include <iterator>
#include <iostream>
#include <optional>
#include <atomic>
#include <chrono>
#include <tuple>
#include <array>
#include <immintrin.h>
#include <windows.h>
#include <random>
#include <set>
#include <thread>
#include <cstdlib>
#include <cstdint>
#include <charconv>
#include <iomanip>
#include <utility>
#include <io.h>
#include <numeric>
#include <list>
#include <sstream>
#include <string>
#include <map>
#include <shared_mutex>
#include <initializer_list>
#include <cassert>
#include <unordered_map>
#include <deque>
#include <emmintrin.h>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <unordered_set>
#include <sys/stat.h>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <memory>
#include <cerrno>
#include <vector>
#include <unistd.h>
#include <sys/mman.h>


// ######## end of system include headers ########


// ######## begin of self header files


// begin /Users/syys/CLionProjects/lc0/src/benchmark/backendbench.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class BackendBenchmark {
 public:
  BackendBenchmark() = default;
  void Run();
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/benchmark/backendbench.h
// begin /Users/syys/CLionProjects/lc0/src/utils/bititer.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
#ifdef _MSC_VER
#endif
namespace lczero {
inline unsigned long GetLowestBit(std::uint64_t value) {
#if defined(_MSC_VER) && defined(_WIN64)
  unsigned long result;
  _BitScanForward64(&result, value);
  return result;
#elif defined(_MSC_VER)
  unsigned long result;
  if (value & 0xFFFFFFFF) {
    _BitScanForward(&result, value);
  } else {
    _BitScanForward(&result, value >> 32);
    result += 32;
  }
  return result;
#else
  return __builtin_ctzll(value);
#endif
}
enum BoardTransform {
  NoTransform = 0,
  // Horizontal mirror, ReverseBitsInBytes
  FlipTransform = 1,
  // Vertical mirror, ReverseBytesInBytes
  MirrorTransform = 2,
  // Diagonal transpose A1 to H8, TransposeBitsInBytes.
  TransposeTransform = 4,
};
inline uint64_t ReverseBitsInBytes(uint64_t v) {
  v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
  v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
  return v;
}
inline uint64_t ReverseBytesInBytes(uint64_t v) {
  v = (v & 0x00000000FFFFFFFF) << 32 | (v & 0xFFFFFFFF00000000) >> 32;
  v = (v & 0x0000FFFF0000FFFF) << 16 | (v & 0xFFFF0000FFFF0000) >> 16;
  v = (v & 0x00FF00FF00FF00FF) << 8 | (v & 0xFF00FF00FF00FF00) >> 8;
  return v;
}
// Transpose across the diagonal connecting bit 7 to bit 56.
inline uint64_t TransposeBitsInBytes(uint64_t v) {
  v = (v & 0xAA00AA00AA00AA00ULL) >> 9 | (v & 0x0055005500550055ULL) << 9 |
      (v & 0x55AA55AA55AA55AAULL);
  v = (v & 0xCCCC0000CCCC0000ULL) >> 18 | (v & 0x0000333300003333ULL) << 18 |
      (v & 0x3333CCCC3333CCCCULL);
  v = (v & 0xF0F0F0F000000000ULL) >> 36 | (v & 0x000000000F0F0F0FULL) << 36 |
      (v & 0x0F0F0F0FF0F0F0F0ULL);
  return v;
}
// Iterates over all set bits of the value, lower to upper. The value of
// dereferenced iterator is bit number (lower to upper, 0 bazed)
template <typename T>
class BitIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = T;
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  BitIterator(std::uint64_t value) : value_(value){};
  bool operator!=(const BitIterator& other) { return value_ != other.value_; }
  void operator++() { value_ &= (value_ - 1); }
  T operator*() const { return GetLowestBit(value_); }
 private:
  std::uint64_t value_;
};
class IterateBits {
 public:
  IterateBits(std::uint64_t value) : value_(value) {}
  BitIterator<int> begin() { return value_; }
  BitIterator<int> end() { return 0; }
 private:
  std::uint64_t value_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/bititer.h
// begin /Users/syys/CLionProjects/lc0/src/chess/bitboard.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Stores a coordinates of a single square.
class BoardSquare {
 public:
  constexpr BoardSquare() {}
  // As a single number, 0 to 63, bottom to top, left to right.
  // 0 is a1, 8 is a2, 63 is h8.
  constexpr BoardSquare(std::uint8_t num) : square_(num) {}
  // From row(bottom to top), and col(left to right), 0-based.
  constexpr BoardSquare(int row, int col) : BoardSquare(row * 8 + col) {}
  // From Square name, e.g e4. Only lowercase.
  BoardSquare(const std::string& str, bool black = false)
      : BoardSquare(black ? '8' - str[1] : str[1] - '1', str[0] - 'a') {}
  constexpr std::uint8_t as_int() const { return square_; }
  constexpr std::uint64_t as_board() const { return 1ULL << square_; }
  void set(int row, int col) { square_ = row * 8 + col; }
  // 0-based, bottom to top.
  int row() const { return square_ / 8; }
  // 0-based, left to right.
  int col() const { return square_ % 8; }
  // Row := 7 - row.  Col remains the same.
  void Mirror() { square_ = square_ ^ 0b111000; }
  // Checks whether coordinate is within 0..7.
  static bool IsValidCoord(int x) { return x >= 0 && x < 8; }
  // Checks whether coordinates are within 0..7.
  static bool IsValid(int row, int col) {
    return row >= 0 && col >= 0 && row < 8 && col < 8;
  }
  constexpr bool operator==(const BoardSquare& other) const {
    return square_ == other.square_;
  }
  constexpr bool operator!=(const BoardSquare& other) const {
    return square_ != other.square_;
  }
  // Returns the square in algebraic notation (e.g. "e4").
  std::string as_string() const {
    return std::string(1, 'a' + col()) + std::string(1, '1' + row());
  }
 private:
  std::uint8_t square_ = 0;  // Only lower six bits should be set.
};
// Represents a board as an array of 64 bits.
// Bit enumeration goes from bottom to top, from left to right:
// Square a1 is bit 0, square a8 is bit 7, square b1 is bit 8.
class BitBoard {
 public:
  constexpr BitBoard(std::uint64_t board) : board_(board) {}
  BitBoard() = default;
  BitBoard(const BitBoard&) = default;
  std::uint64_t as_int() const { return board_; }
  void clear() { board_ = 0; }
  // Counts the number of set bits in the BitBoard.
  int count() const {
#if defined(NO_POPCNT)
    std::uint64_t x = board_;
    x -= (x >> 1) & 0x5555555555555555;
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
    return (x * 0x0101010101010101) >> 56;
#elif defined(_MSC_VER) && defined(_WIN64)
    return _mm_popcnt_u64(board_);
#elif defined(_MSC_VER)
    return __popcnt(board_) + __popcnt(board_ >> 32);
#else
    return __builtin_popcountll(board_);
#endif
  }
  // Like count() but using algorithm faster on a very sparse BitBoard.
  // May be slower for more than 4 set bits, but still correct.
  // Useful when counting bits in a Q, R, N or B BitBoard.
  int count_few() const {
#if defined(NO_POPCNT)
    std::uint64_t x = board_;
    int count;
    for (count = 0; x != 0; ++count) {
      // Clear the rightmost set bit.
      x &= x - 1;
    }
    return count;
#else
    return count();
#endif
  }
  // Sets the value for given square to 1 if cond is true.
  // Otherwise does nothing (doesn't reset!).
  void set_if(BoardSquare square, bool cond) { set_if(square.as_int(), cond); }
  void set_if(std::uint8_t pos, bool cond) {
    board_ |= (std::uint64_t(cond) << pos);
  }
  void set_if(int row, int col, bool cond) {
    set_if(BoardSquare(row, col), cond);
  }
  // Sets value of given square to 1.
  void set(BoardSquare square) { set(square.as_int()); }
  void set(std::uint8_t pos) { board_ |= (std::uint64_t(1) << pos); }
  void set(int row, int col) { set(BoardSquare(row, col)); }
  // Sets value of given square to 0.
  void reset(BoardSquare square) { reset(square.as_int()); }
  void reset(std::uint8_t pos) { board_ &= ~(std::uint64_t(1) << pos); }
  void reset(int row, int col) { reset(BoardSquare(row, col)); }
  // Gets value of a square.
  bool get(BoardSquare square) const { return get(square.as_int()); }
  bool get(std::uint8_t pos) const {
    return board_ & (std::uint64_t(1) << pos);
  }
  bool get(int row, int col) const { return get(BoardSquare(row, col)); }
  // Returns whether all bits of a board are set to 0.
  bool empty() const { return board_ == 0; }
  // Checks whether two bitboards have common bits set.
  bool intersects(const BitBoard& other) const { return board_ & other.board_; }
  // Flips black and white side of a board.
  void Mirror() { board_ = ReverseBytesInBytes(board_); }
  bool operator==(const BitBoard& other) const {
    return board_ == other.board_;
  }
  bool operator!=(const BitBoard& other) const {
    return board_ != other.board_;
  }
  BitIterator<BoardSquare> begin() const { return board_; }
  BitIterator<BoardSquare> end() const { return 0; }
  std::string DebugString() const {
    std::string res;
    for (int i = 7; i >= 0; --i) {
      for (int j = 0; j < 8; ++j) {
        if (get(i, j))
          res += '#';
        else
          res += '.';
      }
      res += '\n';
    }
    return res;
  }
  // Applies a mask to the bitboard (intersects).
  BitBoard& operator&=(const BitBoard& a) {
    board_ &= a.board_;
    return *this;
  }
  friend void swap(BitBoard& a, BitBoard& b) {
    using std::swap;
    swap(a.board_, b.board_);
  }
  // Returns union (bitwise OR) of two boards.
  friend BitBoard operator|(const BitBoard& a, const BitBoard& b) {
    return {a.board_ | b.board_};
  }
  // Returns intersection (bitwise AND) of two boards.
  friend BitBoard operator&(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & b.board_};
  }
  // Returns bitboard with one bit reset.
  friend BitBoard operator-(const BitBoard& a, const BoardSquare& b) {
    return {a.board_ & ~b.as_board()};
  }
  // Returns difference (bitwise AND-NOT) of two boards.
  friend BitBoard operator-(const BitBoard& a, const BitBoard& b) {
    return {a.board_ & ~b.board_};
  }
 private:
  std::uint64_t board_ = 0;
};
class Move {
 public:
  enum class Promotion : std::uint8_t { None, Queen, Rook, Bishop, Knight };
  Move() = default;
  constexpr Move(BoardSquare from, BoardSquare to)
      : data_(to.as_int() + (from.as_int() << 6)) {}
  constexpr Move(BoardSquare from, BoardSquare to, Promotion promotion)
      : data_(to.as_int() + (from.as_int() << 6) +
              (static_cast<uint8_t>(promotion) << 12)) {}
  Move(const std::string& str, bool black = false);
  Move(const char* str, bool black = false) : Move(std::string(str), black) {}
  BoardSquare to() const { return BoardSquare(data_ & kToMask); }
  BoardSquare from() const { return BoardSquare((data_ & kFromMask) >> 6); }
  Promotion promotion() const { return Promotion((data_ & kPromoMask) >> 12); }
  void SetTo(BoardSquare to) { data_ = (data_ & ~kToMask) | to.as_int(); }
  void SetFrom(BoardSquare from) {
    data_ = (data_ & ~kFromMask) | (from.as_int() << 6);
  }
  void SetPromotion(Promotion promotion) {
    data_ = (data_ & ~kPromoMask) | (static_cast<uint8_t>(promotion) << 12);
  }
  // 0 .. 16384, knight promotion and no promotion is the same.
  uint16_t as_packed_int() const;
  // 0 .. 1857, to use in neural networks.
  // Transform is a bit field which describes a transform to be applied to the
  // the move before converting it to an index.
  uint16_t as_nn_index(int transform) const;
  explicit operator bool() const { return data_ != 0; }
  bool operator==(const Move& other) const { return data_ == other.data_; }
  void Mirror() { data_ ^= 0b111000111000; }
  std::string as_string() const {
    std::string res = from().as_string() + to().as_string();
    switch (promotion()) {
      case Promotion::None:
        return res;
      case Promotion::Queen:
        return res + 'q';
      case Promotion::Rook:
        return res + 'r';
      case Promotion::Bishop:
        return res + 'b';
      case Promotion::Knight:
        return res + 'n';
    }
    assert(false);
    return "Error!";
  }
 private:
  uint16_t data_ = 0;
  // Move, using the following encoding:
  // bits 0..5 "to"-square
  // bits 6..11 "from"-square
  // bits 12..14 promotion value
  enum Masks : uint16_t {
    kToMask = 0b0000000000111111,
    kFromMask = 0b0000111111000000,
    kPromoMask = 0b0111000000000000,
  };
};
using MoveList = std::vector<Move>;
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/bitboard.h
// begin /Users/syys/CLionProjects/lc0/src/utils/hashcat.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Tries to scramble @val.
inline uint64_t Hash(uint64_t val) {
  return 0xfad0d7f2fbb059f1ULL * (val + 0xbaad41cdcb839961ULL) +
         0x7acec0050bf82f43ULL * ((val >> 31) + 0xd571b3a92b1b2755ULL);
}
// Appends value to a hash.
inline uint64_t HashCat(uint64_t hash, uint64_t x) {
  hash ^= 0x299799adf0d95defULL + Hash(x) + (hash << 6) + (hash >> 2);
  return hash;
}
// Combines 64-bit values into concatenated hash.
inline uint64_t HashCat(std::initializer_list<uint64_t> args) {
  uint64_t hash = 0;
  for (uint64_t x : args) hash = HashCat(hash, x);
  return hash;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/hashcat.h
// begin /Users/syys/CLionProjects/lc0/src/chess/board.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Initializes internal magic bitboard structures.
void InitializeMagicBitboards();
// Represents king attack info used during legal move detection.
class KingAttackInfo {
 public:
  bool in_check() const { return attack_lines_.as_int(); }
  bool in_double_check() const { return double_check_; }
  bool is_pinned(const BoardSquare square) const {
    return pinned_pieces_.get(square);
  }
  bool is_on_attack_line(const BoardSquare square) const {
    return attack_lines_.get(square);
  }
  bool double_check_ = 0;
  BitBoard pinned_pieces_ = {0};
  BitBoard attack_lines_ = {0};
};
// Represents a board position.
// Unlike most chess engines, the board is mirrored for black.
class ChessBoard {
 public:
  ChessBoard() = default;
  ChessBoard(const std::string& fen) { SetFromFen(fen); }
  static const char* kStartposFen;
  static const ChessBoard kStartposBoard;
  static const BitBoard kPawnMask;
  // Sets position from FEN string.
  // If @rule50_ply and @moves are not nullptr, they are filled with number
  // of moves without capture and number of full moves since the beginning of
  // the game.
  void SetFromFen(std::string fen, int* rule50_ply = nullptr,
                  int* moves = nullptr);
  // Nullifies the whole structure.
  void Clear();
  // Swaps black and white pieces and mirrors them relative to the
  // middle of the board. (what was on rank 1 appears on rank 8, what was
  // on file b remains on file b).
  void Mirror();
  // Generates list of possible moves for "ours" (white), but may leave king
  // under check.
  MoveList GeneratePseudolegalMoves() const;
  // Applies the move. (Only for "ours" (white)). Returns true if 50 moves
  // counter should be removed.
  bool ApplyMove(Move move);
  // Checks if the square is under attack from "theirs" (black).
  bool IsUnderAttack(BoardSquare square) const;
  // Generates the king attack info used for legal move detection.
  KingAttackInfo GenerateKingAttackInfo() const;
  // Checks if "our" (white) king is under check.
  bool IsUnderCheck() const { return IsUnderAttack(our_king_); }
  // Checks whether at least one of the sides has mating material.
  bool HasMatingMaterial() const;
  // Generates legal moves.
  MoveList GenerateLegalMoves() const;
  // Check whether pseudolegal move is legal.
  bool IsLegalMove(Move move, const KingAttackInfo& king_attack_info) const;
  // Returns whether two moves are actually the same move in the position.
  bool IsSameMove(Move move1, Move move2) const;
  // Returns the same move but with castling encoded in legacy way.
  Move GetLegacyMove(Move move) const;
  // Returns the same move but with castling encoded in modern way.
  Move GetModernMove(Move move) const;
  uint64_t Hash() const {
    return HashCat({our_pieces_.as_int(), their_pieces_.as_int(),
                    rooks_.as_int(), bishops_.as_int(), pawns_.as_int(),
                    (static_cast<uint32_t>(our_king_.as_int()) << 24) |
                        (static_cast<uint32_t>(their_king_.as_int()) << 16) |
                        (static_cast<uint32_t>(castlings_.as_int()) << 8) |
                        static_cast<uint32_t>(flipped_)});
  }
  class Castlings {
   public:
    Castlings() : queenside_rook_(0), kingside_rook_(7) {}
    void set_we_can_00() { data_ |= 1; }
    void set_we_can_000() { data_ |= 2; }
    void set_they_can_00() { data_ |= 4; }
    void set_they_can_000() { data_ |= 8; }
    void reset_we_can_00() { data_ &= ~1; }
    void reset_we_can_000() { data_ &= ~2; }
    void reset_they_can_00() { data_ &= ~4; }
    void reset_they_can_000() { data_ &= ~8; }
    bool we_can_00() const { return data_ & 1; }
    bool we_can_000() const { return data_ & 2; }
    bool they_can_00() const { return data_ & 4; }
    bool they_can_000() const { return data_ & 8; }
    bool no_legal_castle() const { return data_ == 0; }
    void Mirror() { data_ = ((data_ & 0b11) << 2) + ((data_ & 0b1100) >> 2); }
    // Note: this is not a strict xfen compatible output. Without access to the
    // board its not possible to know whether there is ambiguity so all cases
    // with any non-standard rook positions are encoded in the x-fen format
    std::string as_string() const {
      if (data_ == 0) return "-";
      std::string result;
      if (queenside_rook() == FILE_A && kingside_rook() == FILE_H) {
        if (we_can_00()) result += 'K';
        if (we_can_000()) result += 'Q';
        if (they_can_00()) result += 'k';
        if (they_can_000()) result += 'q';
      } else {
        if (we_can_00()) result += 'A' + kingside_rook();
        if (we_can_000()) result += 'A' + queenside_rook();
        if (they_can_00()) result += 'a' + kingside_rook();
        if (they_can_000()) result += 'a' + queenside_rook();
      }
      return result;
    }
    std::string DebugString() const {
      std::string result;
      if (data_ == 0) result = "-";
      if (we_can_00()) result += 'K';
      if (we_can_000()) result += 'Q';
      if (they_can_00()) result += 'k';
      if (they_can_000()) result += 'q';
      result += '[';
      result += 'a' + queenside_rook();
      result += 'a' + kingside_rook();
      result += ']';
      return result;
    }
    uint8_t as_int() const { return data_; }
    bool operator==(const Castlings& other) const {
      assert(queenside_rook_ == other.queenside_rook_ &&
             kingside_rook_ == other.kingside_rook_);
      return data_ == other.data_;
    }
    uint8_t queenside_rook() const { return queenside_rook_; }
    uint8_t kingside_rook() const { return kingside_rook_; }
    void SetRookPositions(std::uint8_t left, std::uint8_t right) {
      queenside_rook_ = left;
      kingside_rook_ = right;
    }
   private:
    // Position of "left" (queenside) rook in starting game position.
    std::uint8_t queenside_rook_ : 3;
    // Position of "right" (kingside) rook in starting position.
    std::uint8_t kingside_rook_ : 3;
    // - Bit 0 -- "our" side's kingside castle.
    // - Bit 1 -- "our" side's queenside castle.
    // - Bit 2 -- opponent's side's kingside castle.
    // - Bit 3 -- opponent's side's queenside castle.
    std::uint8_t data_ = 0;
  };
  std::string DebugString() const;
  BitBoard ours() const { return our_pieces_; }
  BitBoard theirs() const { return their_pieces_; }
  BitBoard pawns() const { return pawns_ & kPawnMask; }
  BitBoard en_passant() const { return pawns_ - kPawnMask; }
  BitBoard bishops() const { return bishops_ - rooks_; }
  BitBoard rooks() const { return rooks_ - bishops_; }
  BitBoard queens() const { return rooks_ & bishops_; }
  BitBoard knights() const {
    return (our_pieces_ | their_pieces_) - pawns() - our_king_ - their_king_ -
           rooks_ - bishops_;
  }
  BitBoard kings() const {
    return our_king_.as_board() | their_king_.as_board();
  }
  const Castlings& castlings() const { return castlings_; }
  bool flipped() const { return flipped_; }
  bool operator==(const ChessBoard& other) const {
    return (our_pieces_ == other.our_pieces_) &&
           (their_pieces_ == other.their_pieces_) && (rooks_ == other.rooks_) &&
           (bishops_ == other.bishops_) && (pawns_ == other.pawns_) &&
           (our_king_ == other.our_king_) &&
           (their_king_ == other.their_king_) &&
           (castlings_ == other.castlings_) && (flipped_ == other.flipped_);
  }
  bool operator!=(const ChessBoard& other) const { return !operator==(other); }
  enum Square : uint8_t {
    // clang-format off
    A1 = 0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    // clang-format on
  };
  enum File : uint8_t {
    // clang-format off
    FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H
    // clang-format on
  };
  enum Rank : uint8_t {
    // clang-format off
    RANK_1 = 0, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8
    // clang-format on
  };
 private:
  // All white pieces.
  BitBoard our_pieces_;
  // All black pieces.
  BitBoard their_pieces_;
  // Rooks and queens.
  BitBoard rooks_;
  // Bishops and queens;
  BitBoard bishops_;
  // Pawns.
  // Ranks 1 and 8 have special meaning. Pawn at rank 1 means that
  // corresponding white pawn on rank 4 can be taken en passant. Rank 8 is the
  // same for black pawns. Those "fake" pawns are not present in our_pieces_ and
  // their_pieces_ bitboards.
  BitBoard pawns_;
  BoardSquare our_king_;
  BoardSquare their_king_;
  Castlings castlings_;
  bool flipped_ = false;  // aka "Black to move".
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/board.h
// begin /Users/syys/CLionProjects/lc0/src/chess/position.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class Position {
 public:
  // From parent position and move.
  Position(const Position& parent, Move m);
  // From particular position.
  Position(const ChessBoard& board, int rule50_ply, int game_ply);
  uint64_t Hash() const;
  bool IsBlackToMove() const { return us_board_.flipped(); }
  // Number of half-moves since beginning of the game.
  int GetGamePly() const { return ply_count_; }
  // How many time the same position appeared in the game before.
  int GetRepetitions() const { return repetitions_; }
  // How many half-moves since the same position appeared in the game before.
  int GetPliesSincePrevRepetition() const { return cycle_length_; }
  // Someone outside that class knows better about repetitions, so they can
  // set it.
  void SetRepetitions(int repetitions, int cycle_length) {
    repetitions_ = repetitions;
    cycle_length_ = cycle_length;
  }
  // Number of ply with no captures and pawn moves.
  int GetRule50Ply() const { return rule50_ply_; }
  // Gets board from the point of view of player to move.
  const ChessBoard& GetBoard() const { return us_board_; }
  // Gets board from the point of view of opponent.
  const ChessBoard& GetThemBoard() const { return them_board_; }
  // Gets board from the point of view of the white player.
  const ChessBoard& GetWhiteBoard() const {
    return us_board_.flipped() ? them_board_ : us_board_;
  };
  std::string DebugString() const;
 private:
  // The board from the point of view of the player to move.
  ChessBoard us_board_;
  // The board from the point of view of opponent.
  ChessBoard them_board_;
  // How many half-moves without capture or pawn move was there.
  int rule50_ply_ = 0;
  // How many repetitions this position had before. For new positions it's 0.
  int repetitions_;
  // How many half-moves since the position was repeated or 0.
  int cycle_length_;
  // number of half-moves since beginning of the game.
  int ply_count_ = 0;
};
// GetFen returns a FEN notation for the position.
std::string GetFen(const Position& pos);
// These are ordered so max() prefers the best result.
enum class GameResult : uint8_t { UNDECIDED, BLACK_WON, DRAW, WHITE_WON };
GameResult operator-(const GameResult& res);
class PositionHistory {
 public:
  PositionHistory() = default;
  PositionHistory(const PositionHistory& other) = default;
  PositionHistory(PositionHistory&& other) = default;
  PositionHistory& operator=(const PositionHistory& other) = default;
  PositionHistory& operator=(PositionHistory&& other) = default;  
  // Returns first position of the game (or fen from which it was initialized).
  const Position& Starting() const { return positions_.front(); }
  // Returns the latest position of the game.
  const Position& Last() const { return positions_.back(); }
  // N-th position of the game, 0-based.
  const Position& GetPositionAt(int idx) const { return positions_[idx]; }
  // Trims position to a given size.
  void Trim(int size) {
    positions_.erase(positions_.begin() + size, positions_.end());
  }
  // Can be used to reduce allocation cost while performing a sequence of moves
  // in succession.
  void Reserve(int size) { positions_.reserve(size); }
  // Number of positions in history.
  int GetLength() const { return positions_.size(); }
  // Resets the position to a given state.
  void Reset(const ChessBoard& board, int rule50_ply, int game_ply);
  // Appends a position to history.
  void Append(Move m);
  // Pops last move from history.
  void Pop() { positions_.pop_back(); }
  // Finds the endgame state (win/lose/draw/nothing) for the last position.
  GameResult ComputeGameResult() const;
  // Returns whether next move is history should be black's.
  bool IsBlackToMove() const { return Last().IsBlackToMove(); }
  // Builds a hash from last X positions.
  uint64_t HashLast(int positions) const;
  // Checks for any repetitions since the last time 50 move rule was reset.
  bool DidRepeatSinceLastZeroingMove() const;
 private:
  int ComputeLastMoveRepetitions(int* cycle_length) const;
  std::vector<Position> positions_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/position.h
// begin /Users/syys/CLionProjects/lc0/src/chess/callbacks.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Is sent when search decides on the best move.
struct BestMoveInfo {
  BestMoveInfo(Move bestmove, Move ponder = Move{})
      : bestmove(bestmove), ponder(ponder) {}
  Move bestmove;
  Move ponder;
  // Those are extensions and not really UCI protocol.
  // 1 if it's "player1", 2 if it's "player2"
  int player = -1;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player, if known.
  std::optional<bool> is_black;
};
// Is sent during the search.
struct ThinkingInfo {
  // Full depth.
  int depth = -1;
  // Maximum depth.
  int seldepth = -1;
  // Time since start of thinking.
  int64_t time = -1;
  // Nodes visited.
  int64_t nodes = -1;
  // Nodes per second.
  int nps = -1;
  // Hash fullness * 1000
  int hashfull = -1;
  // Moves to mate.
  std::optional<int> mate;
  // Win in centipawns.
  std::optional<int> score;
  // Win/Draw/Lose probability * 1000.
  struct WDL {
    int w;
    int d;
    int l;
  };
  std::optional<WDL> wdl;
  // Number of successful TB probes (not the same as playouts ending in TB hit).
  int tb_hits = -1;
  // Best line found. Moves are from perspective of white player.
  std::vector<Move> pv;
  // Multipv index.
  int multipv = -1;
  // Freeform comment.
  std::string comment;
  // Those are extensions and not really UCI protocol.
  // 1 if it's "player1", 2 if it's "player2"
  int player = -1;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player, if known.
  std::optional<bool> is_black;
  // Moves left
  std::optional<int> moves_left;
};
// Is sent when a single game is finished.
struct GameInfo {
  // Game result.
  GameResult game_result = GameResult::UNDECIDED;
  // Name of the file with training data.
  std::string training_filename;
  // Initial fen of the game.
  std::string initial_fen;
  // Game moves.
  std::vector<Move> moves;
  // Ply within moves that the game actually started.
  int play_start_ply;
  // Index of the game in the tournament (0-based).
  int game_id = -1;
  // The color of the player1, if known.
  std::optional<bool> is_black;
  // Minimum resign threshold which would have resulted in a false positive
  // if resign had of been enabled.
  // Only provided if the game wasn't played with resign enabled.
  std::optional<float> min_false_positive_threshold;
  using Callback = std::function<void(const GameInfo&)>;
};
// Is sent in the end of tournament and also during the tournament.
struct TournamentInfo {
  // Did tournament finish, so those results are final.
  bool finished = false;
  // Player1's [win/draw/lose] as [white/black].
  // e.g. results[2][1] is how many times player 1 lost as black.
  int results[3][2] = {{0, 0}, {0, 0}, {0, 0}};
  int move_count_ = 0;
  uint64_t nodes_total_ = 0;
  using Callback = std::function<void(const TournamentInfo&)>;
};
// A class which knows how to output UCI responses.
class UciResponder {
 public:
  virtual ~UciResponder() = default;
  virtual void OutputBestMove(BestMoveInfo* info) = 0;
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) = 0;
};
// The responder which calls callbacks. Used for easier transition from old
// code.
class CallbackUciResponder : public UciResponder {
 public:
  using ThinkingCallback =
      std::function<void(const std::vector<ThinkingInfo>&)>;
  using BestMoveCallback = std::function<void(const BestMoveInfo&)>;
  CallbackUciResponder(BestMoveCallback bestmove, ThinkingCallback info)
      : bestmove_callback_(bestmove), info_callback_(info) {}
  void OutputBestMove(BestMoveInfo* info) { bestmove_callback_(*info); }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
    info_callback_(*infos);
  }
 private:
  const BestMoveCallback bestmove_callback_;
  const ThinkingCallback info_callback_;
};
// The responnder which doesn't own the parent. Used to transition from old code
// where we need to create a copy.
class NonOwningUciRespondForwarder : public UciResponder {
 public:
  NonOwningUciRespondForwarder(UciResponder* parent) : parent_(parent) {}
  virtual void OutputBestMove(BestMoveInfo* info) {
    parent_->OutputBestMove(info);
  }
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
    parent_->OutputThinkingInfo(infos);
  }
 private:
  UciResponder* const parent_;
};
// Base class for uci response transformations.
class TransformingUciResponder : public UciResponder {
 public:
  TransformingUciResponder(std::unique_ptr<UciResponder> parent)
      : parent_(std::move(parent)) {}
  virtual void TransformBestMove(BestMoveInfo*) {}
  virtual void TransformThinkingInfo(std::vector<ThinkingInfo>*) {}
 private:
  virtual void OutputBestMove(BestMoveInfo* info) {
    TransformBestMove(info);
    parent_->OutputBestMove(info);
  }
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) {
    TransformThinkingInfo(infos);
    parent_->OutputThinkingInfo(infos);
  }
  std::unique_ptr<UciResponder> parent_;
};
class WDLResponseFilter : public TransformingUciResponder {
  using TransformingUciResponder::TransformingUciResponder;
  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    for (auto& info : *infos) info.wdl.reset();
  }
};
class MovesLeftResponseFilter : public TransformingUciResponder {
  using TransformingUciResponder::TransformingUciResponder;
  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    for (auto& info : *infos) info.moves_left.reset();
  }
};
// Remaps FRC castling to legacy castling.
class Chess960Transformer : public TransformingUciResponder {
 public:
  Chess960Transformer(std::unique_ptr<UciResponder> parent,
                      ChessBoard head_board)
      : TransformingUciResponder(std::move(parent)), head_board_(head_board) {}
 private:
  void TransformBestMove(BestMoveInfo* best_move) override {
    std::vector<Move> moves({best_move->bestmove, best_move->ponder});
    ConvertToLegacyCastling(head_board_, &moves);
    best_move->bestmove = moves[0];
    best_move->ponder = moves[1];
  }
  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    for (auto& x : *infos) ConvertToLegacyCastling(head_board_, &x.pv);
  }
  static void ConvertToLegacyCastling(ChessBoard pos,
                                      std::vector<Move>* moves) {
    for (auto& move : *moves) {
      if (pos.flipped()) move.Mirror();
      move = pos.GetLegacyMove(move);
      pos.ApplyMove(move);
      if (pos.flipped()) move.Mirror();
      pos.Mirror();
    }
  }
  const ChessBoard head_board_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/callbacks.h
// begin /Users/syys/CLionProjects/lc0/src/utils/cppattributes.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
// Enable thread safety attributes only with clang.
// The attributes can be safely erased when compiling with other compilers.
#if defined(__clang__) && (!defined(SWIG))
#define ATTRIBUTE__(x) __attribute__((x))
#else
#define ATTRIBUTE__(x)  // no-op
#endif
#define CAPABILITY(x) ATTRIBUTE__(capability(x))
#define SCOPED_CAPABILITY ATTRIBUTE__(scoped_lockable)
#define GUARDED_BY(x) ATTRIBUTE__(guarded_by(x))
#define PT_GUARDED_BY(x) ATTRIBUTE__(pt_guarded_by(x))
#define ACQUIRED_BEFORE(...) ATTRIBUTE__(acquired_before(__VA_ARGS__))
#define ACQUIRED_AFTER(...) ATTRIBUTE__(acquired_after(__VA_ARGS__))
#define REQUIRES(...) ATTRIBUTE__(requires_capability(__VA_ARGS__))
#define REQUIRES_SHARED(...) \
  ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))
#define ACQUIRE(...) ATTRIBUTE__(acquire_capability(__VA_ARGS__))
#define ACQUIRE_SHARED(...) ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))
#define RELEASE(...) ATTRIBUTE__(release_capability(__VA_ARGS__))
#define RELEASE_SHARED(...) ATTRIBUTE__(release_shared_capability(__VA_ARGS__))
#define TRY_ACQUIRE(...) ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))
#define TRY_ACQUIRE_SHARED(...) \
  ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))
#define EXCLUDES(...) ATTRIBUTE__(locks_excluded(__VA_ARGS__))
#define ASSERT_CAPABILITY(x) ATTRIBUTE__(assert_capability(x))
#define ASSERT_SHARED_CAPABILITY(x) ATTRIBUTE__(assert_shared_capability(x))
#define RETURN_CAPABILITY(x) ATTRIBUTE__(lock_returned(x))
#define PACKED_STRUCT ATTRIBUTE__(packed)
#define NO_THREAD_SAFETY_ANALYSIS ATTRIBUTE__(no_thread_safety_analysis)

// end of /Users/syys/CLionProjects/lc0/src/utils/cppattributes.h
// begin /Users/syys/CLionProjects/lc0/src/utils/mutex.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
#endif
namespace lczero {
// Implementation of reader-preferenced shared mutex. Based on fair shared
// mutex.
class CAPABILITY("mutex") RpSharedMutex {
 public:
  RpSharedMutex() : waiting_readers_(0) {}
  void lock() ACQUIRE() {
    while (true) {
      mutex_.lock();
      if (waiting_readers_ == 0) return;
      mutex_.unlock();
    }
  }
  void unlock() RELEASE() { mutex_.unlock(); }
  void lock_shared() ACQUIRE_SHARED() {
    ++waiting_readers_;
    mutex_.lock_shared();
  }
  void unlock_shared() RELEASE_SHARED() {
    --waiting_readers_;
    mutex_.unlock_shared();
  }
 private:
  std::shared_timed_mutex mutex_;
  std::atomic<int> waiting_readers_;
};
// std::mutex wrapper for clang thread safety annotation.
class CAPABILITY("mutex") Mutex {
 public:
  // std::unique_lock<std::mutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(Mutex& m) ACQUIRE(m) : lock_(m.get_raw()) {}
    ~Lock() RELEASE() {}
    std::unique_lock<std::mutex>& get_raw() { return lock_; }
   private:
    std::unique_lock<std::mutex> lock_;
  };
  void lock() ACQUIRE() { mutex_.lock(); }
  void unlock() RELEASE() { mutex_.unlock(); }
  std::mutex& get_raw() { return mutex_; }
 private:
  std::mutex mutex_;
};
// std::shared_mutex wrapper for clang thread safety annotation.
class CAPABILITY("mutex") SharedMutex {
 public:
  // std::unique_lock<std::shared_mutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(SharedMutex& m) ACQUIRE(m) : lock_(m.get_raw()) {}
    ~Lock() RELEASE() {}
   private:
    std::unique_lock<std::shared_timed_mutex> lock_;
  };
  // std::shared_lock<std::shared_mutex> wrapper.
  class SCOPED_CAPABILITY SharedLock {
   public:
    SharedLock(SharedMutex& m) ACQUIRE_SHARED(m) : lock_(m.get_raw()) {}
    ~SharedLock() RELEASE() {}
   private:
    std::shared_lock<std::shared_timed_mutex> lock_;
  };
  void lock() ACQUIRE() { mutex_.lock(); }
  void unlock() RELEASE() { mutex_.unlock(); }
  void lock_shared() ACQUIRE_SHARED() { mutex_.lock_shared(); }
  void unlock_shared() RELEASE_SHARED() { mutex_.unlock_shared(); }
  std::shared_timed_mutex& get_raw() { return mutex_; }
 private:
  std::shared_timed_mutex mutex_;
};
static inline void SpinloopPause() {
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
  _mm_pause();
#endif
}
// A very simple spin lock.
class CAPABILITY("mutex") SpinMutex {
 public:
  // std::unique_lock<SpinMutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(SpinMutex& m) ACQUIRE(m) : lock_(m) {}
    ~Lock() RELEASE() {}
   private:
    std::unique_lock<SpinMutex> lock_;
  };
  void lock() ACQUIRE() {
    int spins = 0;
    while (true) {
      int val = 0;
      if (mutex_.compare_exchange_weak(val, 1, std::memory_order_acq_rel)) {
        break;
      }
      ++spins;
      // Help avoid complete resource starvation by yielding occasionally if
      // needed.
      if (spins % 512 == 0) {
        std::this_thread::yield();
      } else {
        SpinloopPause();
      }
    }
  }
  void unlock() RELEASE() { mutex_.store(0, std::memory_order_release); }
 private:
  std::atomic<int> mutex_{0};
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/mutex.h
// begin /Users/syys/CLionProjects/lc0/src/utils/logging.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class Logging {
 public:
  static Logging& Get();
  // Sets the name of the log. Empty name disables logging.
  void SetFilename(const std::string& filename);
 private:
  // Writes line to the log, and appends new line character.
  void WriteLineRaw(const std::string& line);
  Mutex mutex_;
  std::string filename_ GUARDED_BY(mutex_);
  std::ofstream file_ GUARDED_BY(mutex_);
  std::deque<std::string> buffer_ GUARDED_BY(mutex_);
  Logging() = default;
  friend class LogMessage;
};
class LogMessage : public std::ostringstream {
 public:
  LogMessage(const char* file, int line);
  ~LogMessage();
};
class StderrLogMessage : public std::ostringstream {
 public:
  StderrLogMessage(const char* file, int line);
  ~StderrLogMessage();
 private:
  LogMessage log_;
};
class StdoutLogMessage : public std::ostringstream {
 public:
  StdoutLogMessage(const char* file, int line);
  ~StdoutLogMessage();
 private:
  LogMessage log_;
};
std::chrono::time_point<std::chrono::system_clock> SteadyClockToSystemClock(
    std::chrono::time_point<std::chrono::steady_clock> time);
std::string FormatTime(std::chrono::time_point<std::chrono::system_clock> time);
}  // namespace lczero
#define LOGFILE ::lczero::LogMessage(__FILE__, __LINE__)
#define CERR ::lczero::StderrLogMessage(__FILE__, __LINE__)
#define COUT ::lczero::StdoutLogMessage(__FILE__, __LINE__)
// end of /Users/syys/CLionProjects/lc0/src/utils/logging.h
// begin /Users/syys/CLionProjects/lc0/src/utils/exception.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Exception to throw around.
class Exception : public std::runtime_error {
 public:
  Exception(const std::string& what) : std::runtime_error(what) {
    LOGFILE << "Exception: " << what;
  }
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/exception.h
// begin /Users/syys/CLionProjects/lc0/src/chess/uciloop.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
struct GoParams {
  std::optional<std::int64_t> wtime;
  std::optional<std::int64_t> btime;
  std::optional<std::int64_t> winc;
  std::optional<std::int64_t> binc;
  std::optional<int> movestogo;
  std::optional<int> depth;
  std::optional<int> nodes;
  std::optional<std::int64_t> movetime;
  bool infinite = false;
  std::vector<std::string> searchmoves;
  bool ponder = false;
};
class UciLoop {
 public:
  virtual ~UciLoop() {}
  virtual void RunLoop();
  // Sends response to host.
  void SendResponse(const std::string& response);
  // Sends responses to host ensuring they are received as a block.
  virtual void SendResponses(const std::vector<std::string>& responses);
  void SendBestMove(const BestMoveInfo& move);
  void SendInfo(const std::vector<ThinkingInfo>& infos);
  void SendId();
  // Command handlers.
  virtual void CmdUci() { throw Exception("Not supported"); }
  virtual void CmdIsReady() { throw Exception("Not supported"); }
  virtual void CmdSetOption(const std::string& /*name*/,
                            const std::string& /*value*/,
                            const std::string& /*context*/) {
    throw Exception("Not supported");
  }
  virtual void CmdUciNewGame() { throw Exception("Not supported"); }
  virtual void CmdPosition(const std::string& /*position*/,
                           const std::vector<std::string>& /*moves*/) {
    throw Exception("Not supported");
  }
  virtual void CmdFen() { throw Exception("Not supported"); }
  virtual void CmdGo(const GoParams& /*params*/) {
    throw Exception("Not supported");
  }
  virtual void CmdStop() { throw Exception("Not supported"); }
  virtual void CmdPonderHit() { throw Exception("Not supported"); }
  virtual void CmdStart() { throw Exception("Not supported"); }
 private:
  bool DispatchCommand(
      const std::string& command,
      const std::unordered_map<std::string, std::string>& params);
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/uciloop.h
// begin /Users/syys/CLionProjects/lc0/src/utils/protomessage.h
#pragma once
// Undef g++ macros to ged rid of warnings.
#ifdef minor
#undef minor
#endif
#ifdef major
#undef major
#endif
namespace lczero {
class ProtoMessage {
 public:
  virtual ~ProtoMessage() {}
  virtual void Clear() = 0;
  void ParseFromString(std::string_view);
  void MergeFromString(std::string_view);
  virtual std::string OutputAsString() const = 0;
 protected:
  template <class To, class From>
  static To bit_cast(From from) {
    if constexpr (std::is_same_v<From, To>) {
      return from;
    } else {
      To to;
      std::memcpy(&to, &from, sizeof(to));
      return to;
    }
  }
  void AppendVarInt(int field_id, std::uint64_t value, std::string* out) const;
  void AppendInt64(int field_id, std::uint64_t value, std::string* out) const;
  void AppendInt32(int field_id, std::uint32_t value, std::string* out) const;
  void AppendString(int field_id, std::string_view value,
                    std::string* out) const;
 private:
  virtual void SetVarInt(int /* field_id */, uint64_t /* value */) {}
  virtual void SetInt64(int /* field_id */, uint64_t /* value */) {}
  virtual void SetInt32(int /* field_id */, uint32_t /* value */) {}
  virtual void SetString(int /* field_id */, std::string_view /* value */) {}
};
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/utils/protomessage.h
// begin /Users/syys/CLionProjects/lc0/botzone/proto/net.pb.h
// This file is AUTOGENERATED, do not edit.
#pragma once
namespace pblczero {
  class EngineVersion : public lczero::ProtoMessage {
   public:
    bool has_major() const { return has_major_; }
    std::uint32_t major() const { return major_; }
    void set_major(std::uint32_t val) {
      has_major_ = true;
      major_ = val;
    }
    bool has_minor() const { return has_minor_; }
    std::uint32_t minor() const { return minor_; }
    void set_minor(std::uint32_t val) {
      has_minor_ = true;
      minor_ = val;
    }
    bool has_patch() const { return has_patch_; }
    std::uint32_t patch() const { return patch_; }
    void set_patch(std::uint32_t val) {
      has_patch_ = true;
      patch_ = val;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_major_) AppendVarInt(1, major_, &out);
      if (has_minor_) AppendVarInt(2, minor_, &out);
      if (has_patch_) AppendVarInt(3, patch_, &out);
      return out;
    }
    void Clear() override {
      has_major_ = false;
      major_ = {};
      has_minor_ = false;
      minor_ = {};
      has_patch_ = false;
      patch_ = {};
    }
   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_major(static_cast<std::uint32_t>(val)); break;
        case 2: set_minor(static_cast<std::uint32_t>(val)); break;
        case 3: set_patch(static_cast<std::uint32_t>(val)); break;
      }
    }
    bool has_major_{};
    std::uint32_t major_{};
    bool has_minor_{};
    std::uint32_t minor_{};
    bool has_patch_{};
    std::uint32_t patch_{};
  };
  class Weights : public lczero::ProtoMessage {
   public:
    class Layer : public lczero::ProtoMessage {
     public:
      bool has_min_val() const { return has_min_val_; }
      float min_val() const { return min_val_; }
      void set_min_val(float val) {
        has_min_val_ = true;
        min_val_ = val;
      }
      bool has_max_val() const { return has_max_val_; }
      float max_val() const { return max_val_; }
      void set_max_val(float val) {
        has_max_val_ = true;
        max_val_ = val;
      }
      bool has_params() const { return has_params_; }
      std::string_view params() const { return params_; }
      void set_params(std::string_view val) {
        has_params_ = true;
        params_ = val;
      }
      std::string OutputAsString() const override {
        std::string out;
        if (has_min_val_) AppendInt32(1, bit_cast<std::uint32_t>(min_val_), &out);
        if (has_max_val_) AppendInt32(2, bit_cast<std::uint32_t>(max_val_), &out);
        if (has_params_) AppendString(3, params_, &out);
        return out;
      }
      void Clear() override {
        has_min_val_ = false;
        min_val_ = {};
        has_max_val_ = false;
        max_val_ = {};
        has_params_ = false;
        params_ = {};
      }
     private:
      void SetInt32(int field_id, std::uint32_t val) override {
        switch (field_id) {
          case 1: set_min_val(bit_cast<float>(val)); break;
          case 2: set_max_val(bit_cast<float>(val)); break;
        }
      }
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 3: set_params(val); break;
        }
      }
      bool has_min_val_{};
      float min_val_{};
      bool has_max_val_{};
      float max_val_{};
      bool has_params_{};
      std::string params_{};
    };
    class ConvBlock : public lczero::ProtoMessage {
     public:
      bool has_weights() const { return has_weights_; }
      const Layer& weights() const { return weights_; }
      Layer* mutable_weights() {
        has_weights_ = true;
        return &weights_;
      }
      bool has_biases() const { return has_biases_; }
      const Layer& biases() const { return biases_; }
      Layer* mutable_biases() {
        has_biases_ = true;
        return &biases_;
      }
      bool has_bn_means() const { return has_bn_means_; }
      const Layer& bn_means() const { return bn_means_; }
      Layer* mutable_bn_means() {
        has_bn_means_ = true;
        return &bn_means_;
      }
      bool has_bn_stddivs() const { return has_bn_stddivs_; }
      const Layer& bn_stddivs() const { return bn_stddivs_; }
      Layer* mutable_bn_stddivs() {
        has_bn_stddivs_ = true;
        return &bn_stddivs_;
      }
      bool has_bn_gammas() const { return has_bn_gammas_; }
      const Layer& bn_gammas() const { return bn_gammas_; }
      Layer* mutable_bn_gammas() {
        has_bn_gammas_ = true;
        return &bn_gammas_;
      }
      bool has_bn_betas() const { return has_bn_betas_; }
      const Layer& bn_betas() const { return bn_betas_; }
      Layer* mutable_bn_betas() {
        has_bn_betas_ = true;
        return &bn_betas_;
      }
      std::string OutputAsString() const override {
        std::string out;
        if (has_weights_) AppendString(1, weights_.OutputAsString(), &out);
        if (has_biases_) AppendString(2, biases_.OutputAsString(), &out);
        if (has_bn_means_) AppendString(3, bn_means_.OutputAsString(), &out);
        if (has_bn_stddivs_) AppendString(4, bn_stddivs_.OutputAsString(), &out);
        if (has_bn_gammas_) AppendString(5, bn_gammas_.OutputAsString(), &out);
        if (has_bn_betas_) AppendString(6, bn_betas_.OutputAsString(), &out);
        return out;
      }
      void Clear() override {
        has_weights_ = false;
        weights_ = {};
        has_biases_ = false;
        biases_ = {};
        has_bn_means_ = false;
        bn_means_ = {};
        has_bn_stddivs_ = false;
        bn_stddivs_ = {};
        has_bn_gammas_ = false;
        bn_gammas_ = {};
        has_bn_betas_ = false;
        bn_betas_ = {};
      }
     private:
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 1: mutable_weights()->MergeFromString(val); break;
          case 2: mutable_biases()->MergeFromString(val); break;
          case 3: mutable_bn_means()->MergeFromString(val); break;
          case 4: mutable_bn_stddivs()->MergeFromString(val); break;
          case 5: mutable_bn_gammas()->MergeFromString(val); break;
          case 6: mutable_bn_betas()->MergeFromString(val); break;
        }
      }
      bool has_weights_{};
      Layer weights_{};
      bool has_biases_{};
      Layer biases_{};
      bool has_bn_means_{};
      Layer bn_means_{};
      bool has_bn_stddivs_{};
      Layer bn_stddivs_{};
      bool has_bn_gammas_{};
      Layer bn_gammas_{};
      bool has_bn_betas_{};
      Layer bn_betas_{};
    };
    class SEunit : public lczero::ProtoMessage {
     public:
      bool has_w1() const { return has_w1_; }
      const Layer& w1() const { return w1_; }
      Layer* mutable_w1() {
        has_w1_ = true;
        return &w1_;
      }
      bool has_b1() const { return has_b1_; }
      const Layer& b1() const { return b1_; }
      Layer* mutable_b1() {
        has_b1_ = true;
        return &b1_;
      }
      bool has_w2() const { return has_w2_; }
      const Layer& w2() const { return w2_; }
      Layer* mutable_w2() {
        has_w2_ = true;
        return &w2_;
      }
      bool has_b2() const { return has_b2_; }
      const Layer& b2() const { return b2_; }
      Layer* mutable_b2() {
        has_b2_ = true;
        return &b2_;
      }
      std::string OutputAsString() const override {
        std::string out;
        if (has_w1_) AppendString(1, w1_.OutputAsString(), &out);
        if (has_b1_) AppendString(2, b1_.OutputAsString(), &out);
        if (has_w2_) AppendString(3, w2_.OutputAsString(), &out);
        if (has_b2_) AppendString(4, b2_.OutputAsString(), &out);
        return out;
      }
      void Clear() override {
        has_w1_ = false;
        w1_ = {};
        has_b1_ = false;
        b1_ = {};
        has_w2_ = false;
        w2_ = {};
        has_b2_ = false;
        b2_ = {};
      }
     private:
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 1: mutable_w1()->MergeFromString(val); break;
          case 2: mutable_b1()->MergeFromString(val); break;
          case 3: mutable_w2()->MergeFromString(val); break;
          case 4: mutable_b2()->MergeFromString(val); break;
        }
      }
      bool has_w1_{};
      Layer w1_{};
      bool has_b1_{};
      Layer b1_{};
      bool has_w2_{};
      Layer w2_{};
      bool has_b2_{};
      Layer b2_{};
    };
    class Residual : public lczero::ProtoMessage {
     public:
      bool has_conv1() const { return has_conv1_; }
      const ConvBlock& conv1() const { return conv1_; }
      ConvBlock* mutable_conv1() {
        has_conv1_ = true;
        return &conv1_;
      }
      bool has_conv2() const { return has_conv2_; }
      const ConvBlock& conv2() const { return conv2_; }
      ConvBlock* mutable_conv2() {
        has_conv2_ = true;
        return &conv2_;
      }
      bool has_se() const { return has_se_; }
      const SEunit& se() const { return se_; }
      SEunit* mutable_se() {
        has_se_ = true;
        return &se_;
      }
      std::string OutputAsString() const override {
        std::string out;
        if (has_conv1_) AppendString(1, conv1_.OutputAsString(), &out);
        if (has_conv2_) AppendString(2, conv2_.OutputAsString(), &out);
        if (has_se_) AppendString(3, se_.OutputAsString(), &out);
        return out;
      }
      void Clear() override {
        has_conv1_ = false;
        conv1_ = {};
        has_conv2_ = false;
        conv2_ = {};
        has_se_ = false;
        se_ = {};
      }
     private:
      void SetString(int field_id, std::string_view val) override {
        switch (field_id) {
          case 1: mutable_conv1()->MergeFromString(val); break;
          case 2: mutable_conv2()->MergeFromString(val); break;
          case 3: mutable_se()->MergeFromString(val); break;
        }
      }
      bool has_conv1_{};
      ConvBlock conv1_{};
      bool has_conv2_{};
      ConvBlock conv2_{};
      bool has_se_{};
      SEunit se_{};
    };
    bool has_input() const { return has_input_; }
    const ConvBlock& input() const { return input_; }
    ConvBlock* mutable_input() {
      has_input_ = true;
      return &input_;
    }
    Residual* add_residual() { return &residual_.emplace_back(); }
    const std::vector<Residual>& residual() const { return residual_; }
    const Residual& residual(size_t idx) const { return residual_[idx]; }
    size_t residual_size() const { return residual_.size(); }
    bool has_policy1() const { return has_policy1_; }
    const ConvBlock& policy1() const { return policy1_; }
    ConvBlock* mutable_policy1() {
      has_policy1_ = true;
      return &policy1_;
    }
    bool has_policy() const { return has_policy_; }
    const ConvBlock& policy() const { return policy_; }
    ConvBlock* mutable_policy() {
      has_policy_ = true;
      return &policy_;
    }
    bool has_ip_pol_w() const { return has_ip_pol_w_; }
    const Layer& ip_pol_w() const { return ip_pol_w_; }
    Layer* mutable_ip_pol_w() {
      has_ip_pol_w_ = true;
      return &ip_pol_w_;
    }
    bool has_ip_pol_b() const { return has_ip_pol_b_; }
    const Layer& ip_pol_b() const { return ip_pol_b_; }
    Layer* mutable_ip_pol_b() {
      has_ip_pol_b_ = true;
      return &ip_pol_b_;
    }
    bool has_value() const { return has_value_; }
    const ConvBlock& value() const { return value_; }
    ConvBlock* mutable_value() {
      has_value_ = true;
      return &value_;
    }
    bool has_ip1_val_w() const { return has_ip1_val_w_; }
    const Layer& ip1_val_w() const { return ip1_val_w_; }
    Layer* mutable_ip1_val_w() {
      has_ip1_val_w_ = true;
      return &ip1_val_w_;
    }
    bool has_ip1_val_b() const { return has_ip1_val_b_; }
    const Layer& ip1_val_b() const { return ip1_val_b_; }
    Layer* mutable_ip1_val_b() {
      has_ip1_val_b_ = true;
      return &ip1_val_b_;
    }
    bool has_ip2_val_w() const { return has_ip2_val_w_; }
    const Layer& ip2_val_w() const { return ip2_val_w_; }
    Layer* mutable_ip2_val_w() {
      has_ip2_val_w_ = true;
      return &ip2_val_w_;
    }
    bool has_ip2_val_b() const { return has_ip2_val_b_; }
    const Layer& ip2_val_b() const { return ip2_val_b_; }
    Layer* mutable_ip2_val_b() {
      has_ip2_val_b_ = true;
      return &ip2_val_b_;
    }
    bool has_moves_left() const { return has_moves_left_; }
    const ConvBlock& moves_left() const { return moves_left_; }
    ConvBlock* mutable_moves_left() {
      has_moves_left_ = true;
      return &moves_left_;
    }
    bool has_ip1_mov_w() const { return has_ip1_mov_w_; }
    const Layer& ip1_mov_w() const { return ip1_mov_w_; }
    Layer* mutable_ip1_mov_w() {
      has_ip1_mov_w_ = true;
      return &ip1_mov_w_;
    }
    bool has_ip1_mov_b() const { return has_ip1_mov_b_; }
    const Layer& ip1_mov_b() const { return ip1_mov_b_; }
    Layer* mutable_ip1_mov_b() {
      has_ip1_mov_b_ = true;
      return &ip1_mov_b_;
    }
    bool has_ip2_mov_w() const { return has_ip2_mov_w_; }
    const Layer& ip2_mov_w() const { return ip2_mov_w_; }
    Layer* mutable_ip2_mov_w() {
      has_ip2_mov_w_ = true;
      return &ip2_mov_w_;
    }
    bool has_ip2_mov_b() const { return has_ip2_mov_b_; }
    const Layer& ip2_mov_b() const { return ip2_mov_b_; }
    Layer* mutable_ip2_mov_b() {
      has_ip2_mov_b_ = true;
      return &ip2_mov_b_;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_input_) AppendString(1, input_.OutputAsString(), &out);
      for (const auto& x : residual_) AppendString(2, x.OutputAsString(), &out);
      if (has_policy_) AppendString(3, policy_.OutputAsString(), &out);
      if (has_ip_pol_w_) AppendString(4, ip_pol_w_.OutputAsString(), &out);
      if (has_ip_pol_b_) AppendString(5, ip_pol_b_.OutputAsString(), &out);
      if (has_value_) AppendString(6, value_.OutputAsString(), &out);
      if (has_ip1_val_w_) AppendString(7, ip1_val_w_.OutputAsString(), &out);
      if (has_ip1_val_b_) AppendString(8, ip1_val_b_.OutputAsString(), &out);
      if (has_ip2_val_w_) AppendString(9, ip2_val_w_.OutputAsString(), &out);
      if (has_ip2_val_b_) AppendString(10, ip2_val_b_.OutputAsString(), &out);
      if (has_policy1_) AppendString(11, policy1_.OutputAsString(), &out);
      if (has_moves_left_) AppendString(12, moves_left_.OutputAsString(), &out);
      if (has_ip1_mov_w_) AppendString(13, ip1_mov_w_.OutputAsString(), &out);
      if (has_ip1_mov_b_) AppendString(14, ip1_mov_b_.OutputAsString(), &out);
      if (has_ip2_mov_w_) AppendString(15, ip2_mov_w_.OutputAsString(), &out);
      if (has_ip2_mov_b_) AppendString(16, ip2_mov_b_.OutputAsString(), &out);
      return out;
    }
    void Clear() override {
      has_input_ = false;
      input_ = {};
      residual_.clear();
      has_policy1_ = false;
      policy1_ = {};
      has_policy_ = false;
      policy_ = {};
      has_ip_pol_w_ = false;
      ip_pol_w_ = {};
      has_ip_pol_b_ = false;
      ip_pol_b_ = {};
      has_value_ = false;
      value_ = {};
      has_ip1_val_w_ = false;
      ip1_val_w_ = {};
      has_ip1_val_b_ = false;
      ip1_val_b_ = {};
      has_ip2_val_w_ = false;
      ip2_val_w_ = {};
      has_ip2_val_b_ = false;
      ip2_val_b_ = {};
      has_moves_left_ = false;
      moves_left_ = {};
      has_ip1_mov_w_ = false;
      ip1_mov_w_ = {};
      has_ip1_mov_b_ = false;
      ip1_mov_b_ = {};
      has_ip2_mov_w_ = false;
      ip2_mov_w_ = {};
      has_ip2_mov_b_ = false;
      ip2_mov_b_ = {};
    }
   private:
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 1: mutable_input()->MergeFromString(val); break;
        case 2: add_residual()->MergeFromString(val); break;
        case 11: mutable_policy1()->MergeFromString(val); break;
        case 3: mutable_policy()->MergeFromString(val); break;
        case 4: mutable_ip_pol_w()->MergeFromString(val); break;
        case 5: mutable_ip_pol_b()->MergeFromString(val); break;
        case 6: mutable_value()->MergeFromString(val); break;
        case 7: mutable_ip1_val_w()->MergeFromString(val); break;
        case 8: mutable_ip1_val_b()->MergeFromString(val); break;
        case 9: mutable_ip2_val_w()->MergeFromString(val); break;
        case 10: mutable_ip2_val_b()->MergeFromString(val); break;
        case 12: mutable_moves_left()->MergeFromString(val); break;
        case 13: mutable_ip1_mov_w()->MergeFromString(val); break;
        case 14: mutable_ip1_mov_b()->MergeFromString(val); break;
        case 15: mutable_ip2_mov_w()->MergeFromString(val); break;
        case 16: mutable_ip2_mov_b()->MergeFromString(val); break;
      }
    }
    bool has_input_{};
    ConvBlock input_{};
    std::vector<Residual> residual_;
    bool has_policy1_{};
    ConvBlock policy1_{};
    bool has_policy_{};
    ConvBlock policy_{};
    bool has_ip_pol_w_{};
    Layer ip_pol_w_{};
    bool has_ip_pol_b_{};
    Layer ip_pol_b_{};
    bool has_value_{};
    ConvBlock value_{};
    bool has_ip1_val_w_{};
    Layer ip1_val_w_{};
    bool has_ip1_val_b_{};
    Layer ip1_val_b_{};
    bool has_ip2_val_w_{};
    Layer ip2_val_w_{};
    bool has_ip2_val_b_{};
    Layer ip2_val_b_{};
    bool has_moves_left_{};
    ConvBlock moves_left_{};
    bool has_ip1_mov_w_{};
    Layer ip1_mov_w_{};
    bool has_ip1_mov_b_{};
    Layer ip1_mov_b_{};
    bool has_ip2_mov_w_{};
    Layer ip2_mov_w_{};
    bool has_ip2_mov_b_{};
    Layer ip2_mov_b_{};
  };
  class TrainingParams : public lczero::ProtoMessage {
   public:
    bool has_training_steps() const { return has_training_steps_; }
    std::uint32_t training_steps() const { return training_steps_; }
    void set_training_steps(std::uint32_t val) {
      has_training_steps_ = true;
      training_steps_ = val;
    }
    bool has_learning_rate() const { return has_learning_rate_; }
    float learning_rate() const { return learning_rate_; }
    void set_learning_rate(float val) {
      has_learning_rate_ = true;
      learning_rate_ = val;
    }
    bool has_mse_loss() const { return has_mse_loss_; }
    float mse_loss() const { return mse_loss_; }
    void set_mse_loss(float val) {
      has_mse_loss_ = true;
      mse_loss_ = val;
    }
    bool has_policy_loss() const { return has_policy_loss_; }
    float policy_loss() const { return policy_loss_; }
    void set_policy_loss(float val) {
      has_policy_loss_ = true;
      policy_loss_ = val;
    }
    bool has_accuracy() const { return has_accuracy_; }
    float accuracy() const { return accuracy_; }
    void set_accuracy(float val) {
      has_accuracy_ = true;
      accuracy_ = val;
    }
    bool has_lc0_params() const { return has_lc0_params_; }
    std::string_view lc0_params() const { return lc0_params_; }
    void set_lc0_params(std::string_view val) {
      has_lc0_params_ = true;
      lc0_params_ = val;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_training_steps_) AppendVarInt(1, training_steps_, &out);
      if (has_learning_rate_) AppendInt32(2, bit_cast<std::uint32_t>(learning_rate_), &out);
      if (has_mse_loss_) AppendInt32(3, bit_cast<std::uint32_t>(mse_loss_), &out);
      if (has_policy_loss_) AppendInt32(4, bit_cast<std::uint32_t>(policy_loss_), &out);
      if (has_accuracy_) AppendInt32(5, bit_cast<std::uint32_t>(accuracy_), &out);
      if (has_lc0_params_) AppendString(6, lc0_params_, &out);
      return out;
    }
    void Clear() override {
      has_training_steps_ = false;
      training_steps_ = {};
      has_learning_rate_ = false;
      learning_rate_ = {};
      has_mse_loss_ = false;
      mse_loss_ = {};
      has_policy_loss_ = false;
      policy_loss_ = {};
      has_accuracy_ = false;
      accuracy_ = {};
      has_lc0_params_ = false;
      lc0_params_ = {};
    }
   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_training_steps(static_cast<std::uint32_t>(val)); break;
      }
    }
    void SetInt32(int field_id, std::uint32_t val) override {
      switch (field_id) {
        case 2: set_learning_rate(bit_cast<float>(val)); break;
        case 3: set_mse_loss(bit_cast<float>(val)); break;
        case 4: set_policy_loss(bit_cast<float>(val)); break;
        case 5: set_accuracy(bit_cast<float>(val)); break;
      }
    }
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 6: set_lc0_params(val); break;
      }
    }
    bool has_training_steps_{};
    std::uint32_t training_steps_{};
    bool has_learning_rate_{};
    float learning_rate_{};
    bool has_mse_loss_{};
    float mse_loss_{};
    bool has_policy_loss_{};
    float policy_loss_{};
    bool has_accuracy_{};
    float accuracy_{};
    bool has_lc0_params_{};
    std::string lc0_params_{};
  };
  class NetworkFormat : public lczero::ProtoMessage {
   public:
    enum InputFormat {
      INPUT_UNKNOWN = 0,
      INPUT_CLASSICAL_112_PLANE = 1,
      INPUT_112_WITH_CASTLING_PLANE = 2,
      INPUT_112_WITH_CANONICALIZATION = 3,
      INPUT_112_WITH_CANONICALIZATION_HECTOPLIES = 4,
      INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON = 132,
      INPUT_112_WITH_CANONICALIZATION_V2 = 5,
      INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON = 133,
    };
    static constexpr std::array<InputFormat,8> InputFormat_AllValues = {
      INPUT_UNKNOWN,
      INPUT_CLASSICAL_112_PLANE,
      INPUT_112_WITH_CASTLING_PLANE,
      INPUT_112_WITH_CANONICALIZATION,
      INPUT_112_WITH_CANONICALIZATION_HECTOPLIES,
      INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON,
      INPUT_112_WITH_CANONICALIZATION_V2,
      INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON,
    };
    static std::string InputFormat_Name(InputFormat val) {
      switch (val) {
        case INPUT_UNKNOWN:
          return "INPUT_UNKNOWN";
        case INPUT_CLASSICAL_112_PLANE:
          return "INPUT_CLASSICAL_112_PLANE";
        case INPUT_112_WITH_CASTLING_PLANE:
          return "INPUT_112_WITH_CASTLING_PLANE";
        case INPUT_112_WITH_CANONICALIZATION:
          return "INPUT_112_WITH_CANONICALIZATION";
        case INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
          return "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES";
        case INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
          return "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON";
        case INPUT_112_WITH_CANONICALIZATION_V2:
          return "INPUT_112_WITH_CANONICALIZATION_V2";
        case INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON:
          return "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON";
      };
      return "InputFormat(" + std::to_string(val) + ")";
    }
    enum OutputFormat {
      OUTPUT_UNKNOWN = 0,
      OUTPUT_CLASSICAL = 1,
      OUTPUT_WDL = 2,
    };
    static constexpr std::array<OutputFormat,3> OutputFormat_AllValues = {
      OUTPUT_UNKNOWN,
      OUTPUT_CLASSICAL,
      OUTPUT_WDL,
    };
    static std::string OutputFormat_Name(OutputFormat val) {
      switch (val) {
        case OUTPUT_UNKNOWN:
          return "OUTPUT_UNKNOWN";
        case OUTPUT_CLASSICAL:
          return "OUTPUT_CLASSICAL";
        case OUTPUT_WDL:
          return "OUTPUT_WDL";
      };
      return "OutputFormat(" + std::to_string(val) + ")";
    }
    enum NetworkStructure {
      NETWORK_UNKNOWN = 0,
      NETWORK_CLASSICAL = 1,
      NETWORK_SE = 2,
      NETWORK_CLASSICAL_WITH_HEADFORMAT = 3,
      NETWORK_SE_WITH_HEADFORMAT = 4,
      NETWORK_ONNX = 5,
    };
    static constexpr std::array<NetworkStructure,6> NetworkStructure_AllValues = {
      NETWORK_UNKNOWN,
      NETWORK_CLASSICAL,
      NETWORK_SE,
      NETWORK_CLASSICAL_WITH_HEADFORMAT,
      NETWORK_SE_WITH_HEADFORMAT,
      NETWORK_ONNX,
    };
    static std::string NetworkStructure_Name(NetworkStructure val) {
      switch (val) {
        case NETWORK_UNKNOWN:
          return "NETWORK_UNKNOWN";
        case NETWORK_CLASSICAL:
          return "NETWORK_CLASSICAL";
        case NETWORK_SE:
          return "NETWORK_SE";
        case NETWORK_CLASSICAL_WITH_HEADFORMAT:
          return "NETWORK_CLASSICAL_WITH_HEADFORMAT";
        case NETWORK_SE_WITH_HEADFORMAT:
          return "NETWORK_SE_WITH_HEADFORMAT";
        case NETWORK_ONNX:
          return "NETWORK_ONNX";
      };
      return "NetworkStructure(" + std::to_string(val) + ")";
    }
    enum PolicyFormat {
      POLICY_UNKNOWN = 0,
      POLICY_CLASSICAL = 1,
      POLICY_CONVOLUTION = 2,
    };
    static constexpr std::array<PolicyFormat,3> PolicyFormat_AllValues = {
      POLICY_UNKNOWN,
      POLICY_CLASSICAL,
      POLICY_CONVOLUTION,
    };
    static std::string PolicyFormat_Name(PolicyFormat val) {
      switch (val) {
        case POLICY_UNKNOWN:
          return "POLICY_UNKNOWN";
        case POLICY_CLASSICAL:
          return "POLICY_CLASSICAL";
        case POLICY_CONVOLUTION:
          return "POLICY_CONVOLUTION";
      };
      return "PolicyFormat(" + std::to_string(val) + ")";
    }
    enum ValueFormat {
      VALUE_UNKNOWN = 0,
      VALUE_CLASSICAL = 1,
      VALUE_WDL = 2,
      VALUE_PARAM = 3,
    };
    static constexpr std::array<ValueFormat,4> ValueFormat_AllValues = {
      VALUE_UNKNOWN,
      VALUE_CLASSICAL,
      VALUE_WDL,
      VALUE_PARAM,
    };
    static std::string ValueFormat_Name(ValueFormat val) {
      switch (val) {
        case VALUE_UNKNOWN:
          return "VALUE_UNKNOWN";
        case VALUE_CLASSICAL:
          return "VALUE_CLASSICAL";
        case VALUE_WDL:
          return "VALUE_WDL";
        case VALUE_PARAM:
          return "VALUE_PARAM";
      };
      return "ValueFormat(" + std::to_string(val) + ")";
    }
    enum MovesLeftFormat {
      MOVES_LEFT_NONE = 0,
      MOVES_LEFT_V1 = 1,
    };
    static constexpr std::array<MovesLeftFormat,2> MovesLeftFormat_AllValues = {
      MOVES_LEFT_NONE,
      MOVES_LEFT_V1,
    };
    static std::string MovesLeftFormat_Name(MovesLeftFormat val) {
      switch (val) {
        case MOVES_LEFT_NONE:
          return "MOVES_LEFT_NONE";
        case MOVES_LEFT_V1:
          return "MOVES_LEFT_V1";
      };
      return "MovesLeftFormat(" + std::to_string(val) + ")";
    }
    bool has_input() const { return has_input_; }
    InputFormat input() const { return input_; }
    void set_input(InputFormat val) {
      has_input_ = true;
      input_ = val;
    }
    bool has_output() const { return has_output_; }
    OutputFormat output() const { return output_; }
    void set_output(OutputFormat val) {
      has_output_ = true;
      output_ = val;
    }
    bool has_network() const { return has_network_; }
    NetworkStructure network() const { return network_; }
    void set_network(NetworkStructure val) {
      has_network_ = true;
      network_ = val;
    }
    bool has_policy() const { return has_policy_; }
    PolicyFormat policy() const { return policy_; }
    void set_policy(PolicyFormat val) {
      has_policy_ = true;
      policy_ = val;
    }
    bool has_value() const { return has_value_; }
    ValueFormat value() const { return value_; }
    void set_value(ValueFormat val) {
      has_value_ = true;
      value_ = val;
    }
    bool has_moves_left() const { return has_moves_left_; }
    MovesLeftFormat moves_left() const { return moves_left_; }
    void set_moves_left(MovesLeftFormat val) {
      has_moves_left_ = true;
      moves_left_ = val;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_input_) AppendVarInt(1, input_, &out);
      if (has_output_) AppendVarInt(2, output_, &out);
      if (has_network_) AppendVarInt(3, network_, &out);
      if (has_policy_) AppendVarInt(4, policy_, &out);
      if (has_value_) AppendVarInt(5, value_, &out);
      if (has_moves_left_) AppendVarInt(6, moves_left_, &out);
      return out;
    }
    void Clear() override {
      has_input_ = false;
      input_ = {};
      has_output_ = false;
      output_ = {};
      has_network_ = false;
      network_ = {};
      has_policy_ = false;
      policy_ = {};
      has_value_ = false;
      value_ = {};
      has_moves_left_ = false;
      moves_left_ = {};
    }
   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_input(static_cast<InputFormat>(val)); break;
        case 2: set_output(static_cast<OutputFormat>(val)); break;
        case 3: set_network(static_cast<NetworkStructure>(val)); break;
        case 4: set_policy(static_cast<PolicyFormat>(val)); break;
        case 5: set_value(static_cast<ValueFormat>(val)); break;
        case 6: set_moves_left(static_cast<MovesLeftFormat>(val)); break;
      }
    }
    bool has_input_{};
    InputFormat input_{};
    bool has_output_{};
    OutputFormat output_{};
    bool has_network_{};
    NetworkStructure network_{};
    bool has_policy_{};
    PolicyFormat policy_{};
    bool has_value_{};
    ValueFormat value_{};
    bool has_moves_left_{};
    MovesLeftFormat moves_left_{};
  };
  class Format : public lczero::ProtoMessage {
   public:
    enum Encoding {
      UNKNOWN = 0,
      LINEAR16 = 1,
    };
    static constexpr std::array<Encoding,2> Encoding_AllValues = {
      UNKNOWN,
      LINEAR16,
    };
    static std::string Encoding_Name(Encoding val) {
      switch (val) {
        case UNKNOWN:
          return "UNKNOWN";
        case LINEAR16:
          return "LINEAR16";
      };
      return "Encoding(" + std::to_string(val) + ")";
    }
    bool has_weights_encoding() const { return has_weights_encoding_; }
    Encoding weights_encoding() const { return weights_encoding_; }
    void set_weights_encoding(Encoding val) {
      has_weights_encoding_ = true;
      weights_encoding_ = val;
    }
    bool has_network_format() const { return has_network_format_; }
    const NetworkFormat& network_format() const { return network_format_; }
    NetworkFormat* mutable_network_format() {
      has_network_format_ = true;
      return &network_format_;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_weights_encoding_) AppendVarInt(1, weights_encoding_, &out);
      if (has_network_format_) AppendString(2, network_format_.OutputAsString(), &out);
      return out;
    }
    void Clear() override {
      has_weights_encoding_ = false;
      weights_encoding_ = {};
      has_network_format_ = false;
      network_format_ = {};
    }
   private:
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 1: set_weights_encoding(static_cast<Encoding>(val)); break;
      }
    }
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 2: mutable_network_format()->MergeFromString(val); break;
      }
    }
    bool has_weights_encoding_{};
    Encoding weights_encoding_{};
    bool has_network_format_{};
    NetworkFormat network_format_{};
  };
  class OnnxModel : public lczero::ProtoMessage {
   public:
    enum DataType {
      UNKNOWN_DATATYPE = 0,
      FLOAT = 1,
      FLOAT16 = 10,
      BFLOAT16 = 16,
    };
    static constexpr std::array<DataType,4> DataType_AllValues = {
      UNKNOWN_DATATYPE,
      FLOAT,
      FLOAT16,
      BFLOAT16,
    };
    static std::string DataType_Name(DataType val) {
      switch (val) {
        case UNKNOWN_DATATYPE:
          return "UNKNOWN_DATATYPE";
        case FLOAT:
          return "FLOAT";
        case FLOAT16:
          return "FLOAT16";
        case BFLOAT16:
          return "BFLOAT16";
      };
      return "DataType(" + std::to_string(val) + ")";
    }
    bool has_model() const { return has_model_; }
    std::string_view model() const { return model_; }
    void set_model(std::string_view val) {
      has_model_ = true;
      model_ = val;
    }
    bool has_data_type() const { return has_data_type_; }
    DataType data_type() const { return data_type_; }
    void set_data_type(DataType val) {
      has_data_type_ = true;
      data_type_ = val;
    }
    bool has_input_planes() const { return has_input_planes_; }
    std::string_view input_planes() const { return input_planes_; }
    void set_input_planes(std::string_view val) {
      has_input_planes_ = true;
      input_planes_ = val;
    }
    bool has_output_value() const { return has_output_value_; }
    std::string_view output_value() const { return output_value_; }
    void set_output_value(std::string_view val) {
      has_output_value_ = true;
      output_value_ = val;
    }
    bool has_output_wdl() const { return has_output_wdl_; }
    std::string_view output_wdl() const { return output_wdl_; }
    void set_output_wdl(std::string_view val) {
      has_output_wdl_ = true;
      output_wdl_ = val;
    }
    bool has_output_policy() const { return has_output_policy_; }
    std::string_view output_policy() const { return output_policy_; }
    void set_output_policy(std::string_view val) {
      has_output_policy_ = true;
      output_policy_ = val;
    }
    bool has_output_mlh() const { return has_output_mlh_; }
    std::string_view output_mlh() const { return output_mlh_; }
    void set_output_mlh(std::string_view val) {
      has_output_mlh_ = true;
      output_mlh_ = val;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_model_) AppendString(1, model_, &out);
      if (has_data_type_) AppendVarInt(2, data_type_, &out);
      if (has_input_planes_) AppendString(3, input_planes_, &out);
      if (has_output_value_) AppendString(4, output_value_, &out);
      if (has_output_wdl_) AppendString(5, output_wdl_, &out);
      if (has_output_policy_) AppendString(6, output_policy_, &out);
      if (has_output_mlh_) AppendString(7, output_mlh_, &out);
      return out;
    }
    void Clear() override {
      has_model_ = false;
      model_ = {};
      has_data_type_ = false;
      data_type_ = {};
      has_input_planes_ = false;
      input_planes_ = {};
      has_output_value_ = false;
      output_value_ = {};
      has_output_wdl_ = false;
      output_wdl_ = {};
      has_output_policy_ = false;
      output_policy_ = {};
      has_output_mlh_ = false;
      output_mlh_ = {};
    }
   private:
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 1: set_model(val); break;
        case 3: set_input_planes(val); break;
        case 4: set_output_value(val); break;
        case 5: set_output_wdl(val); break;
        case 6: set_output_policy(val); break;
        case 7: set_output_mlh(val); break;
      }
    }
    void SetVarInt(int field_id, std::uint64_t val) override {
      switch (field_id) {
        case 2: set_data_type(static_cast<DataType>(val)); break;
      }
    }
    bool has_model_{};
    std::string model_{};
    bool has_data_type_{};
    DataType data_type_{};
    bool has_input_planes_{};
    std::string input_planes_{};
    bool has_output_value_{};
    std::string output_value_{};
    bool has_output_wdl_{};
    std::string output_wdl_{};
    bool has_output_policy_{};
    std::string output_policy_{};
    bool has_output_mlh_{};
    std::string output_mlh_{};
  };
  class Net : public lczero::ProtoMessage {
   public:
    bool has_magic() const { return has_magic_; }
    std::uint32_t magic() const { return magic_; }
    void set_magic(std::uint32_t val) {
      has_magic_ = true;
      magic_ = val;
    }
    bool has_license() const { return has_license_; }
    std::string_view license() const { return license_; }
    void set_license(std::string_view val) {
      has_license_ = true;
      license_ = val;
    }
    bool has_min_version() const { return has_min_version_; }
    const EngineVersion& min_version() const { return min_version_; }
    EngineVersion* mutable_min_version() {
      has_min_version_ = true;
      return &min_version_;
    }
    bool has_format() const { return has_format_; }
    const Format& format() const { return format_; }
    Format* mutable_format() {
      has_format_ = true;
      return &format_;
    }
    bool has_training_params() const { return has_training_params_; }
    const TrainingParams& training_params() const { return training_params_; }
    TrainingParams* mutable_training_params() {
      has_training_params_ = true;
      return &training_params_;
    }
    bool has_weights() const { return has_weights_; }
    const Weights& weights() const { return weights_; }
    Weights* mutable_weights() {
      has_weights_ = true;
      return &weights_;
    }
    bool has_onnx_model() const { return has_onnx_model_; }
    const OnnxModel& onnx_model() const { return onnx_model_; }
    OnnxModel* mutable_onnx_model() {
      has_onnx_model_ = true;
      return &onnx_model_;
    }
    std::string OutputAsString() const override {
      std::string out;
      if (has_magic_) AppendInt32(1, magic_, &out);
      if (has_license_) AppendString(2, license_, &out);
      if (has_min_version_) AppendString(3, min_version_.OutputAsString(), &out);
      if (has_format_) AppendString(4, format_.OutputAsString(), &out);
      if (has_training_params_) AppendString(5, training_params_.OutputAsString(), &out);
      if (has_weights_) AppendString(10, weights_.OutputAsString(), &out);
      if (has_onnx_model_) AppendString(11, onnx_model_.OutputAsString(), &out);
      return out;
    }
    void Clear() override {
      has_magic_ = false;
      magic_ = {};
      has_license_ = false;
      license_ = {};
      has_min_version_ = false;
      min_version_ = {};
      has_format_ = false;
      format_ = {};
      has_training_params_ = false;
      training_params_ = {};
      has_weights_ = false;
      weights_ = {};
      has_onnx_model_ = false;
      onnx_model_ = {};
    }
   private:
    void SetInt32(int field_id, std::uint32_t val) override {
      switch (field_id) {
        case 1: set_magic(static_cast<std::uint32_t>(val)); break;
      }
    }
    void SetString(int field_id, std::string_view val) override {
      switch (field_id) {
        case 2: set_license(val); break;
        case 3: mutable_min_version()->MergeFromString(val); break;
        case 4: mutable_format()->MergeFromString(val); break;
        case 5: mutable_training_params()->MergeFromString(val); break;
        case 10: mutable_weights()->MergeFromString(val); break;
        case 11: mutable_onnx_model()->MergeFromString(val); break;
      }
    }
    bool has_magic_{};
    std::uint32_t magic_{};
    bool has_license_{};
    std::string license_{};
    bool has_min_version_{};
    EngineVersion min_version_{};
    bool has_format_{};
    Format format_{};
    bool has_training_params_{};
    TrainingParams training_params_{};
    bool has_weights_{};
    Weights weights_{};
    bool has_onnx_model_{};
    OnnxModel onnx_model_{};
  };
}  // namespace pblczero

// end of /Users/syys/CLionProjects/lc0/botzone/proto/net.pb.h
// begin /Users/syys/CLionProjects/lc0/src/neural/network.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
const int kInputPlanes = 112;
// All input planes are 64 value vectors, every element of which is either
// 0 or some value, unique for the plane. Therefore, input is defined as
// a bitmask showing where to set the value, and the value itself.
struct InputPlane {
  InputPlane() = default;
  void SetAll() { mask = ~0ull; }
  void Fill(float val) {
    SetAll();
    value = val;
  }
  std::uint64_t mask = 0ull;
  float value = 1.0f;
};
using InputPlanes = std::vector<InputPlane>;
// An interface to implement by computing backends.
class NetworkComputation {
 public:
  // Adds a sample to the batch.
  virtual void AddInput(InputPlanes&& input) = 0;
  // Do the computation.
  virtual void ComputeBlocking() = 0;
  // Returns how many times AddInput() was called.
  virtual int GetBatchSize() const = 0;
  // Returns Q value of @sample.
  virtual float GetQVal(int sample) const = 0;
  virtual float GetDVal(int sample) const = 0;
  // Returns P value @move_id of @sample.
  virtual float GetPVal(int sample, int move_id) const = 0;
  virtual float GetMVal(int sample) const = 0;
  virtual ~NetworkComputation() = default;
};
// The plan:
// 1. Search must not look directly into any fields of NetworkFormat anymore.
// 2. Backends populate NetworkCapabilities that show search how to use NN, both
//    for input and output.
// 3. Input part of NetworkCapabilities is just copy of InputFormat for now, and
//    is likely to stay so (because search not knowing how to use NN is not very
//    useful), but it's fine if it will change.
// 4. On the other hand, output part of NetworkCapabilities is set of
//    independent parameters (like WDL, moves left head etc), because search can
//    look what's set and act accordingly. Backends may derive it from
//    output head format fields or other places.
struct NetworkCapabilities {
  pblczero::NetworkFormat::InputFormat input_format;
  pblczero::NetworkFormat::MovesLeftFormat moves_left;
  // TODO expose information of whether GetDVal() is usable or always zero.
  // Combines capabilities by setting the most restrictive ones. May throw
  // exception.
  void Merge(const NetworkCapabilities& other) {
    if (input_format != other.input_format) {
      throw Exception("Incompatible input formats, " +
                      std::to_string(input_format) + " vs " +
                      std::to_string(other.input_format));
    }
  }
  bool has_mlh() const {
    return moves_left !=
           pblczero::NetworkFormat::MovesLeftFormat::MOVES_LEFT_NONE;
  }
};
class Network {
 public:
  virtual const NetworkCapabilities& GetCapabilities() const = 0;
  virtual std::unique_ptr<NetworkComputation> NewComputation() = 0;
  virtual ~Network() = default;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/network.h
// begin /Users/syys/CLionProjects/lc0/src/utils/cache.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// A hash-keyed cache. Thread-safe. Takes ownership of all values, which are
// deleted upon eviction; thus, using values stored requires pinning them, which
// in turn requires Unpin()ing them after use. The use of HashKeyedCacheLock is
// recommend to automate this element-memory management.
// Unlike LRUCache, doesn't even consider trying to support LRU order.
// Does not support delete.
// Does not support replace! Inserts to existing elements are silently ignored.
// FIFO eviction.
// Assumes that eviction while pinned is rare enough to not need to optimize
// unpin for that case.
template <class V>
class HashKeyedCache {
  static const double constexpr kLoadFactor = 1.9;
 public:
  HashKeyedCache(int capacity = 128)
      : capacity_(capacity),
        hash_(static_cast<size_t>(capacity * kLoadFactor + 1)) {}
  ~HashKeyedCache() {
    EvictToCapacity(0);
    assert(size_ == 0);
    assert(allocated_ == 0);
  }
  // Inserts the element under key @key with value @val. Unless the key is
  // already in the cache.
  void Insert(uint64_t key, std::unique_ptr<V> val) {
    if (capacity_.load(std::memory_order_relaxed) == 0) return;
    SpinMutex::Lock lock(mutex_);
    size_t idx = key % hash_.size();
    while (true) {
      if (!hash_[idx].in_use) break;
      if (hash_[idx].key == key) {
        // Already exists.
        return;
      }
      ++idx;
      if (idx >= hash_.size()) idx -= hash_.size();
    }
    hash_[idx].key = key;
    hash_[idx].value = std::move(val);
    hash_[idx].pins = 0;
    hash_[idx].in_use = true;
    insertion_order_.push_back(key);
    ++size_;
    ++allocated_;
    EvictToCapacity(capacity_);
  }
  // Checks whether a key exists. Doesn't pin. Of course the next moment the
  // key may be evicted.
  bool ContainsKey(uint64_t key) {
    if (capacity_.load(std::memory_order_relaxed) == 0) return false;
    SpinMutex::Lock lock(mutex_);
    size_t idx = key % hash_.size();
    while (true) {
      if (!hash_[idx].in_use) break;
      if (hash_[idx].key == key) {
        return true;
      }
      ++idx;
      if (idx >= hash_.size()) idx -= hash_.size();
    }
    return false;
  }
  // Looks up and pins the element by key. Returns nullptr if not found.
  // If found, a call to Unpin must be made for each such element.
  // Use of HashedKeyCacheLock is recommended to automate this pin management.
  V* LookupAndPin(uint64_t key) {
    if (capacity_.load(std::memory_order_relaxed) == 0) return nullptr;
    SpinMutex::Lock lock(mutex_);
    size_t idx = key % hash_.size();
    while (true) {
      if (!hash_[idx].in_use) break;
      if (hash_[idx].key == key) {
        ++hash_[idx].pins;
        return hash_[idx].value.get();
      }
      ++idx;
      if (idx >= hash_.size()) idx -= hash_.size();
    }
    return nullptr;
  }
  // Unpins the element given key and value. Use of HashedKeyCacheLock is
  // recommended to automate this pin management.
  void Unpin(uint64_t key, V* value) {
    SpinMutex::Lock lock(mutex_);
    // Checking evicted list first.
    for (auto it = evicted_.begin(); it != evicted_.end(); ++it) {
      auto& entry = *it;
      if (key == entry.key && value == entry.value.get()) {
        if (--entry.pins == 0) {
          --allocated_;
          evicted_.erase(it);
          return;
        } else {
          return;
        }
      }
    }
    // Now the main list.
    size_t idx = key % hash_.size();
    while (true) {
      if (!hash_[idx].in_use) break;
      if (hash_[idx].key == key &&
          hash_[idx].value.get() == value) {
        --hash_[idx].pins;
        return;
      }
      ++idx;
      if (idx >= hash_.size()) idx -= hash_.size();
    }
    assert(false);
  }
  // Sets the capacity of the cache. If new capacity is less than current size
  // of the cache, oldest entries are evicted. In any case the hashtable is
  // rehashed.
  void SetCapacity(int capacity) {
    // This is the one operation that can be expected to take a long time, which
    // usually means a SpinMutex is not a great idea. However we should only
    // very rarely have any contention on the lock while this function is
    // running, since its called very rarely and almost always before things
    // start happening.
    SpinMutex::Lock lock(mutex_);
    if (capacity_.load(std::memory_order_relaxed) == capacity) return;
    EvictToCapacity(capacity);
    capacity_.store(capacity);
    std::vector<Entry> new_hash(
        static_cast<size_t>(capacity * kLoadFactor + 1));
    if (size_ != 0) {
      for (Entry& item : hash_) {
        if (!item.in_use) continue;
        size_t idx = item.key % new_hash.size();
        while (true) {
          if (!new_hash[idx].in_use) break;
          ++idx;
          if (idx >= new_hash.size()) idx -= new_hash.size();
        }
        new_hash[idx].key = item.key;
        new_hash[idx].value = std::move(item.value);
        new_hash[idx].pins = item.pins;
        new_hash[idx].in_use = true;
      }
    }
    hash_.swap(new_hash);
  }
  // Clears the cache;
  void Clear() {
    SpinMutex::Lock lock(mutex_);
    EvictToCapacity(0);
  }
  int GetSize() const {
    SpinMutex::Lock lock(mutex_);
    return size_;
  }
  int GetCapacity() const { return capacity_.load(std::memory_order_relaxed); }
  static constexpr size_t GetItemStructSize() { return sizeof(Entry); }
 private:
  struct Entry {
    Entry() {}
    Entry(uint64_t key, std::unique_ptr<V> value)
        : key(key), value(std::move(value)) {}
    uint64_t key;
    std::unique_ptr<V> value;
    int pins = 0;
    bool in_use = false;
  };
  void EvictItem() REQUIRES(mutex_) {
    --size_;
    uint64_t key = insertion_order_.front();
    insertion_order_.pop_front();
    size_t idx = key % hash_.size();
    while (true) {
      if (hash_[idx].in_use && hash_[idx].key == key) {
        break;
      }
      ++idx;
      if (idx >= hash_.size()) idx -= hash_.size();
    }
    if (hash_[idx].pins == 0) {
      --allocated_;
      hash_[idx].value.reset();
      hash_[idx].in_use = false;
    } else {
      evicted_.emplace_back(hash_[idx].key, std::move(hash_[idx].value));
      evicted_.back().pins = hash_[idx].pins;
      hash_[idx].pins = 0;
      hash_[idx].in_use = false;
    }
    size_t next = idx + 1;
    if (next >= hash_.size()) next -= hash_.size();
    while (true) {
      if (!hash_[next].in_use) {
        break;
      }
      size_t target = hash_[next].key % hash_.size();
      if (!InRange(target, idx + 1, next)) {
        std::swap(hash_[next], hash_[idx]);
        idx = next;
      }
      ++next;
      if (next >= hash_.size()) next -= hash_.size();
    }
  }
  bool InRange(size_t target, size_t start, size_t end) {
    if (start <= end) {
      return target >= start && target <= end;
    } else {
      return target >= start || target <= end;
    }
  }
  void EvictToCapacity(int capacity) REQUIRES(mutex_) {
    if (capacity < 0) capacity = 0;
    while (size_ > capacity) {
      EvictItem();
    }
  }
  std::atomic<int> capacity_;
  int size_ GUARDED_BY(mutex_) = 0;
  int allocated_ GUARDED_BY(mutex_) = 0;
  // Fresh in back, stale at front.
  std::deque<uint64_t> GUARDED_BY(mutex_) insertion_order_;
  std::vector<Entry> GUARDED_BY(mutex_) evicted_;
  std::vector<Entry> GUARDED_BY(mutex_) hash_;
  mutable SpinMutex mutex_;
};
// Convenience class for pinning cache items.
template <class V>
class HashKeyedCacheLock {
 public:
  // Looks up the value in @cache by @key and pins it if found.
  HashKeyedCacheLock(HashKeyedCache<V>* cache, uint64_t key)
      : cache_(cache), key_(key), value_(cache->LookupAndPin(key_)) {}
  // Unpins the cache entry (if holds).
  ~HashKeyedCacheLock() {
    if (value_) cache_->Unpin(key_, value_);
  }
  HashKeyedCacheLock(const HashKeyedCacheLock&) = delete;
  // Returns whether lock holds any value.
  operator bool() const { return value_; }
  // Gets the value.
  V* operator->() const { return value_; }
  V* operator*() const { return value_; }
  HashKeyedCacheLock() {}
  HashKeyedCacheLock(HashKeyedCacheLock&& other)
      : cache_(other.cache_), key_(other.key_), value_(other.value_) {
    other.value_ = nullptr;
  }
  void operator=(HashKeyedCacheLock&& other) {
    if (value_) cache_->Unpin(key_, value_);
    cache_ = other.cache_;
    key_ = other.key_;
    value_ = other.value_;
    other.value_ = nullptr;
  }
 private:
  HashKeyedCache<V>* cache_ = nullptr;
  uint64_t key_;
  V* value_ = nullptr;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/cache.h
// begin /Users/syys/CLionProjects/lc0/src/utils/smallarray.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Non resizeable array which can contain up to 255 elements.
template <typename T>
class SmallArray {
 public:
  SmallArray() = delete;
  SmallArray(size_t size) : size_(size), data_(std::make_unique<T[]>(size)) {}
  SmallArray(SmallArray&&);  // TODO implement when needed
  T& operator[](int idx) { return data_[idx]; }
  const T& operator[](int idx) const { return data_[idx]; }
  int size() const { return size_; }
 private:
  unsigned char size_;
  std::unique_ptr<T[]> data_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/smallarray.h
// begin /Users/syys/CLionProjects/lc0/src/neural/cache.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
struct CachedNNRequest {
  CachedNNRequest(size_t size) : p(size) {}
  typedef std::pair<uint16_t, float> IdxAndProb;
  float q;
  float d;
  float m;
  // TODO(mooskagh) Don't really need index if using perfect hash.
  SmallArray<IdxAndProb> p;
};
typedef HashKeyedCache<CachedNNRequest> NNCache;
typedef HashKeyedCacheLock<CachedNNRequest> NNCacheLock;
// Wraps around NetworkComputation and caches result.
// While it mostly repeats NetworkComputation interface, it's not derived
// from it, as AddInput() needs hash and index of probabilities to store.
class CachingComputation {
 public:
  CachingComputation(std::unique_ptr<NetworkComputation> parent,
                     NNCache* cache);
  // How many inputs are not found in cache and will be forwarded to a wrapped
  // computation.
  int GetCacheMisses() const;
  // Total number of times AddInput/AddInputByHash were (successfully) called.
  int GetBatchSize() const;
  // Adds input by hash only. If that hash is not in cache, returns false
  // and does nothing. Otherwise adds.
  bool AddInputByHash(uint64_t hash);
  // Adds input by hash with existing lock. Assumes the given lock holds a real
  // reference.
  void AddInputByHash(uint64_t hash, NNCacheLock&& lock);
  // Adds a sample to the batch.
  // @hash is a hash to store/lookup it in the cache.
  // @probabilities_to_cache is which indices of policy head to store.
  void AddInput(uint64_t hash, InputPlanes&& input,
                std::vector<uint16_t>&& probabilities_to_cache);
  // Undos last AddInput. If it was a cache miss, the it's actually not removed
  // from parent's batch.
  void PopLastInputHit();
  // Do the computation.
  void ComputeBlocking();
  // Returns Q value of @sample.
  float GetQVal(int sample) const;
  // Returns probability of draw if NN has WDL value head.
  float GetDVal(int sample) const;
  // Returns estimated remaining moves.
  float GetMVal(int sample) const;
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const;
  // Pops last input from the computation. Only allowed for inputs which were
  // cached.
  void PopCacheHit();
  // Can be used to avoid repeated reallocations internally while adding itemms.
  void Reserve(int batch_size) { batch_.reserve(batch_size); }
 private:
  struct WorkItem {
    uint64_t hash;
    NNCacheLock lock;
    int idx_in_parent = -1;
    std::vector<uint16_t> probabilities_to_cache;
    mutable int last_idx = 0;
  };
  std::unique_ptr<NetworkComputation> parent_;
  NNCache* cache_;
  std::vector<WorkItem> batch_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/cache.h
// begin /Users/syys/CLionProjects/lc0/src/neural/encoder.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };
// Returns the transform that would be used in EncodePositionForNN.
int TransformForPosition(pblczero::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history);
// Encodes the last position in history for the neural network request.
InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out);
bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat input_format);
bool IsCanonicalArmageddonFormat(
    pblczero::NetworkFormat::InputFormat input_format);
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat input_format);
bool Is960CastlingFormat(pblczero::NetworkFormat::InputFormat input_format);
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/encoder.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/node.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Children of a node are stored the following way:
// * Edges and Nodes edges point to are stored separately.
// * There may be dangling edges (which don't yet point to any Node object yet)
// * Edges are stored are a simple array on heap.
// * Nodes are stored as a linked list, and contain index_ field which shows
//   which edge of a parent that node points to.
//   Or they are stored a contiguous array of Node objects on the heap if
//   solid_children_ is true. If the children have been 'solidified' their
//   sibling links are unused and left empty. In this state there are no
//   dangling edges, but the nodes may not have ever received any visits.
//
// Example:
//                                Parent Node
//                                    |
//        +-------------+-------------+----------------+--------------+
//        |              |            |                |              |
//   Edge 0(Nf3)    Edge 1(Bc5)     Edge 2(a4)     Edge 3(Qxf7)    Edge 4(a3)
//    (dangling)         |           (dangling)        |           (dangling)
//                   Node, Q=0.5                    Node, Q=-0.2
//
//  Is represented as:
// +--------------+
// | Parent Node  |
// +--------------+                                        +--------+
// | edges_       | -------------------------------------> | Edge[] |
// |              |    +------------+                      +--------+
// | child_       | -> | Node       |                      | Nf3    |
// +--------------+    +------------+                      | Bc5    |
//                     | index_ = 1 |                      | a4     |
//                     | q_ = 0.5   |    +------------+    | Qxf7   |
//                     | sibling_   | -> | Node       |    | a3     |
//                     +------------+    +------------+    +--------+
//                                       | index_ = 3 |
//                                       | q_ = -0.2  |
//                                       | sibling_   | -> nullptr
//                                       +------------+
class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);
  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;
  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);
  // Debug information about the edge.
  std::string DebugString() const;
 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;
  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};
struct Eval {
  float wl;
  float d;
  float ml;
};
class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;
template <bool is_const>
class VisitedNode_Iterator;
class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;
  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };
  // Takes pointer to a parent node and own index in a parent.
  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {}
  // We have a custom destructor, but its behavior does not need to be emulated
  // during move operations so default is fine.
  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;
  // Allocates a new edge and a new node. The node has to be no edges before
  // that.
  Node* CreateSingleChildNode(Move m);
  // Creates edges from a movelist. There has to be no edges before that.
  void CreateEdges(const MoveList& moves);
  // Gets parent node.
  Node* GetParent() const { return parent_; }
  // Returns whether a node has children.
  bool HasChildren() const { return static_cast<bool>(edges_); }
  // Returns sum of policy priors which have had at least one playout.
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 0; }
  // Returns n = n_if_flight.
  int GetNStarted() const { return n_ + n_in_flight_; }
  float GetQ(float draw_score) const { return wl_ + draw_score * d_; }
  // Returns node eval, i.e. average subtree V for non-terminal node and -1/0/1
  // for terminal nodes.
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }
  // Returns whether the node is known to be draw/lose/win.
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint8_t GetNumEdges() const { return num_edges_; }
  // Output must point to at least max_needed floats.
  void CopyPolicy(int max_needed, float* output) const {
    if (!edges_) return;
    int loops = std::min(static_cast<int>(num_edges_), max_needed);
    for (int i = 0; i < loops; i++) {
      output[i] = edges_[i].GetP();
    }
  }
  // Makes the node terminal and sets it's score.
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  // Makes the node not terminal and updates its visits.
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);
  // If this node is not in the process of being expanded by another thread
  // (which can happen only if n==0 and n-in-flight==1), mark the node as
  // "being updated" by incrementing n-in-flight, and return true.
  // Otherwise return false.
  bool TryStartScoreUpdate();
  // Decrements n-in-flight back.
  void CancelScoreUpdate(int multivisit);
  // Updates the node with newly computed value v.
  // Updates:
  // * Q (weighted average of all V in a subtree)
  // * N (+=1)
  // * N-in-flight (-=1)
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  // Like FinalizeScoreUpdate, but it updates n existing visits by delta amount.
  void AdjustForTerminal(float v, float d, float m, int multivisit);
  // Revert visits to a node which ended in a now reverted terminal.
  void RevertTerminalVisits(float v, float d, float m, int multivisit);
  // When search decides to treat one visit as several (in case of collisions
  // or visiting terminal nodes several times), it amplifies the visit by
  // incrementing n_in_flight.
  void IncrementNInFlight(int multivisit) { n_in_flight_ += multivisit; }
  // Updates max depth, if new depth is larger.
  void UpdateMaxDepth(int depth);
  // Calculates the full depth if new depth is larger, updates it, returns
  // in depth parameter, and returns true if it was indeed updated.
  bool UpdateFullDepth(uint16_t* depth);
  // Returns range for iterating over edges.
  ConstIterator Edges() const;
  Iterator Edges();
  // Returns range for iterating over child nodes with N > 0.
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();
  // Deletes all children.
  void ReleaseChildren();
  // Deletes all children except one.
  // The node provided may be moved, so should not be relied upon to exist
  // afterwards.
  void ReleaseChildrenExceptOne(Node* node);
  // For a child node, returns corresponding edge.
  Edge* GetEdgeToNode(const Node* node) const;
  // Returns edge to the own node.
  Edge* GetOwnEdge() const;
  // Debug information about the node.
  std::string DebugString() const;
  // Reallocates this nodes children to be in a solid block, if possible and not
  // already done. Returns true if the transformation was performed.
  bool MakeSolid();
  void SortEdges();
  // Index in parent edges - useful for correlated ordering.
  uint16_t Index() const { return index_; }
  ~Node() {
    if (solid_children_ && child_) {
      // As a hack, solid_children is actually storing an array in here, release
      // so we can correctly invoke the array delete.
      for (int i = 0; i < num_edges_; i++) {
        child_.get()[i].~Node();
      }
      std::allocator<Node> alloc;
      alloc.deallocate(child_.release(), num_edges_);
    }
  }
 private:
  // Performs construction time type initialization. For use only with a node
  // that has not been used beyond its construction.
  void Reinit(Node* parent, uint16_t index) {
    parent_ = parent;
    index_ = index;
  }
  // For each child, ensures that its parent pointer is pointing to this.
  void UpdateChildrenParents();
  // To minimize the number of padding bytes and to avoid having unnecessary
  // padding when new fields are added, we arrange the fields by size, largest
  // to smallest.
  // 8 byte fields.
  // Average value (from value head of neural network) of all visited nodes in
  // subtree. For terminal nodes, eval is stored. This is from the perspective
  // of the player who "just" moved to reach this position, rather than from the
  // perspective of the player-to-move for the position.
  // WL stands for "W minus L". Is equal to Q if draw score is 0.
  double wl_ = 0.0f;
  // 8 byte fields on 64-bit platforms, 4 byte on 32-bit.
  // Array of edges.
  std::unique_ptr<Edge[]> edges_;
  // Pointer to a parent node. nullptr for the root.
  Node* parent_ = nullptr;
  // Pointer to a first child. nullptr for a leaf node.
  // As a 'hack' actually a unique_ptr to Node[] if solid_children.
  std::unique_ptr<Node> child_;
  // Pointer to a next sibling. nullptr if there are no further siblings.
  // Also null in the solid case.
  std::unique_ptr<Node> sibling_;
  // 4 byte fields.
  // Averaged draw probability. Works similarly to WL, except that D is not
  // flipped depending on the side to move.
  float d_ = 0.0f;
  // Estimated remaining plies.
  float m_ = 0.0f;
  // How many completed visits this node had.
  uint32_t n_ = 0;
  // (AKA virtual loss.) How many threads currently process this node (started
  // but not finished). This value is added to n during selection which node
  // to pick in MCTS, and also when selecting the best move.
  uint32_t n_in_flight_ = 0;
  // 2 byte fields.
  // Index of this node is parent's edge list.
  uint16_t index_;
  // 1 byte fields.
  // Number of edges in @edges_.
  uint8_t num_edges_ = 0;
  // Bit fields using parts of uint8_t fields initialized in the constructor.
  // Whether or not this node end game (with a winning of either sides or draw).
  Terminal terminal_type_ : 2;
  // Best and worst result for this node.
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  // Whether the child_ is actually an array of equal length to edges.
  bool solid_children_ : 1;
  // TODO(mooskagh) Unfriend NodeTree.
  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
};
// Define __i386__  or __arm__ also for 32 bit Windows.
#if defined(_M_IX86)
#define __i386__
#endif
#if defined(_M_ARM) && !defined(_M_AMD64)
#define __arm__
#endif
// A basic sanity check. This must be adjusted when Node members are adjusted.
#if defined(__i386__) || (defined(__arm__) && !defined(__aarch64__))
static_assert(sizeof(Node) == 48, "Unexpected size of Node for 32bit compile");
#else
static_assert(sizeof(Node) == 64, "Unexpected size of Node");
#endif
// Contains Edge and Node pair and set of proxy functions to simplify access
// to them.
class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const {
    return edge_ == other.edge_;
  }
  bool operator!=(const EdgeAndNode& other) const {
    return edge_ != other.edge_;
  }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }
  // Proxy functions for easier access to node/edge.
  float GetQ(float default_q, float draw_score) const {
    return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q;
  }
  float GetWL(float default_wl) const {
    return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl;
  }
  float GetD(float default_d) const {
    return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d;
  }
  float GetM(float default_m) const {
    return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m;
  }
  // N-related getters, from Node (if exists).
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
  // Whether the node is known to be terminal.
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Node::Bounds GetBounds() const {
    return node_ ? node_->GetBounds()
                 : Node::Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON};
  }
  // Edge related getters.
  float GetP() const { return edge_->GetP(); }
  Move GetMove(bool flip = false) const {
    return edge_ ? edge_->GetMove(flip) : Move();
  }
  // Returns U = numerator * p / N.
  // Passed numerator is expected to be equal to (cpuct * sqrt(N[parent])).
  float GetU(float numerator) const {
    return numerator * GetP() / (1 + GetNStarted());
  }
  int GetVisitsToReachU(float target_score, float numerator,
                        float score_without_u) const {
    if (score_without_u >= target_score) return std::numeric_limits<int>::max();
    const auto n1 = GetNStarted() + 1;
    return std::max(1.0f,
                    std::min(std::floor(GetP() * numerator /
                                            (target_score - score_without_u) -
                                        n1) +
                                 1,
                             1e9f));
  }
  std::string DebugString() const;
 protected:
  // nullptr means that the whole pair is "null". (E.g. when search for a node
  // didn't find anything, or as end iterator signal).
  Edge* edge_ = nullptr;
  // nullptr means that the edge doesn't yet have node extended.
  Node* node_ = nullptr;
};
// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// * Element, pointed by iterator (EdgeAndNode class mainly, but Edge_Iterator
//   is useful too when client wants to call GetOrSpawnNode).
//   It's safe to slice EdgeAndNode off Edge_Iterator.
// It's more customary to have those as three classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized), but
// it's fine if GetOrSpawnNode is called between calls to functions of the
// iterator (e.g. advancing the iterator). Other functions that manipulate
// child_ of parent or the sibling chain are not safe to call while iterating.
template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*,
                                 std::unique_ptr<Node>*>;
  // Creates "end()" iterator.
  Edge_Iterator() {}
  // Creates "begin()" iterator. Also happens to be a range constructor.
  // child_ptr will be nullptr if parent_node is solid children.
  Edge_Iterator(const Node& parent_node, Ptr child_ptr)
      : EdgeAndNode(parent_node.edges_.get(), nullptr),
        node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_) {
    if (edge_ && child_ptr != nullptr) Actualize();
    if (edge_ && child_ptr == nullptr) {
      node_ = parent_node.child_.get();
    }
  }
  // Function to support range interface.
  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }
  // Functions to support iterator interface.
  // Equality comparison operators are inherited from EdgeAndNode.
  void operator++() {
    // If it was the last edge in array, become end(), otherwise advance.
    if (++current_idx_ == total_count_) {
      edge_ = nullptr;
    } else {
      ++edge_;
      if (node_ptr_ != nullptr) {
        Actualize();
      } else {
        ++node_;
      }
    }
  }
  Edge_Iterator& operator*() { return *this; }
  // If there is node, return it. Otherwise spawn a new one and return it.
  Node* GetOrSpawnNode(Node* parent,
                       std::unique_ptr<Node>* node_source = nullptr) {
    if (node_) return node_;  // If there is already a node, return it.
    // Should never reach here in solid mode.
    assert(node_ptr_ != nullptr);
    Actualize();              // But maybe other thread already did that.
    if (node_) return node_;  // If it did, return.
    // Now we are sure we have to create a new node.
    // Suppose there are nodes with idx 3 and 7, and we want to insert one with
    // idx 5. Here is how it looks like:
    //    node_ptr_ -> &Node(idx_.3).sibling_  ->  Node(idx_.7)
    // Here is how we do that:
    // 1. Store pointer to a node idx_.7:
    //    node_ptr_ -> &Node(idx_.3).sibling_  ->  nullptr
    //    tmp -> Node(idx_.7)
    std::unique_ptr<Node> tmp = std::move(*node_ptr_);
    // 2. Create fresh Node(idx_.5):
    //    node_ptr_ -> &Node(idx_.3).sibling_  ->  Node(idx_.5)
    //    tmp -> Node(idx_.7)
    if (node_source && *node_source) {
      (*node_source)->Reinit(parent, current_idx_);
      *node_ptr_ = std::move(*node_source);
    } else {
      *node_ptr_ = std::make_unique<Node>(parent, current_idx_);
    }
    // 3. Attach stored pointer back to a list:
    //    node_ptr_ ->
    //         &Node(idx_.3).sibling_ -> Node(idx_.5).sibling_ -> Node(idx_.7)
    (*node_ptr_)->sibling_ = std::move(tmp);
    // 4. Actualize:
    //    node_ -> &Node(idx_.5)
    //    node_ptr_ -> &Node(idx_.5).sibling_ -> Node(idx_.7)
    Actualize();
    return node_;
  }
 private:
  void Actualize() {
    // This must never be called in solid mode.
    assert(node_ptr_ != nullptr);
    // If node_ptr_ is behind, advance it.
    // This is needed (and has to be 'while' rather than 'if') as other threads
    // could spawn new nodes between &node_ptr_ and *node_ptr_ while we didn't
    // see.
    while (*node_ptr_ && (*node_ptr_)->index_ < current_idx_) {
      node_ptr_ = &(*node_ptr_)->sibling_;
    }
    // If in the end node_ptr_ points to the node that we need, populate node_
    // and advance node_ptr_.
    if (*node_ptr_ && (*node_ptr_)->index_ == current_idx_) {
      node_ = (*node_ptr_).get();
      node_ptr_ = &node_->sibling_;
    } else {
      node_ = nullptr;
    }
  }
  // Pointer to a pointer to the next node. Has to be a pointer to pointer
  // as we'd like to update it when spawning a new node.
  Ptr node_ptr_;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};
// TODO(crem) Replace this with less hacky iterator once we support C++17.
// This class has multiple hypostases within one class:
// * Range (begin() and end() functions)
// * Iterator (operator++() and operator*())
// It's more customary to have those as two classes, but
// creating zoo of classes and copying them around while iterating seems
// excessive.
//
// All functions are not thread safe (must be externally synchronized).
template <bool is_const>
class VisitedNode_Iterator {
 public:
  // Creates "end()" iterator.
  VisitedNode_Iterator() {}
  // Creates "begin()" iterator. Also happens to be a range constructor.
  // child_ptr will be nullptr if parent_node is solid children.
  VisitedNode_Iterator(const Node& parent_node, Node* child_ptr)
      : node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_),
        solid_(parent_node.solid_children_) {
    if (node_ptr_ != nullptr && node_ptr_->GetN() == 0) {
      operator++();
    }
  }
  // These are technically wrong, but are usable to compare with end().
  bool operator==(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ == other.node_ptr_;
  }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const {
    return node_ptr_ != other.node_ptr_;
  }
  // Function to support range interface.
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }
  // Functions to support iterator interface.
  // Equality comparison operators are inherited from EdgeAndNode.
  void operator++() {
    if (solid_) {
      while (++current_idx_ != total_count_ &&
             node_ptr_[current_idx_].GetN() == 0) {
        if (node_ptr_[current_idx_].GetNInFlight() == 0) {
          // Once there is not even n in flight, we can skip to the end. This is
          // due to policy being in sorted order meaning that additional n in
          // flight are always selected from the front of the section with no n
          // in flight or visited.
          current_idx_ = total_count_;
          break;
        }
      }
      if (current_idx_ == total_count_) {
        node_ptr_ = nullptr;
      }
    } else {
      do {
        node_ptr_ = node_ptr_->sibling_.get();
        // If n started is 0, can jump direct to end due to sorted policy
        // ensuring that each time a new edge becomes best for the first time,
        // it is always the first of the section at the end that has NStarted of
        // 0.
        if (node_ptr_ != nullptr && node_ptr_->GetN() == 0 &&
            node_ptr_->GetNInFlight() == 0) {
          node_ptr_ = nullptr;
          break;
        }
      } while (node_ptr_ != nullptr && node_ptr_->GetN() == 0);
    }
  }
  Node* operator*() {
    if (solid_) {
      return &(node_ptr_[current_idx_]);
    } else {
      return node_ptr_;
    }
  }
 private:
  // Pointer to current node.
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
  bool solid_ = false;
};
inline VisitedNode_Iterator<true> Node::VisitedNodes() const {
  return {*this, child_.get()};
}
inline VisitedNode_Iterator<false> Node::VisitedNodes() {
  return {*this, child_.get()};
}
class NodeTree {
 public:
  ~NodeTree() { DeallocateTree(); }
  // Adds a move to current_head_.
  void MakeMove(Move move);
  // Resets the current head to ensure it doesn't carry over details from a
  // previous search.
  void TrimTreeAtHead();
  // Sets the position in a tree, trying to reuse the tree.
  // If @auto_garbage_collect, old tree is garbage collected immediately. (may
  // take some milliseconds)
  // Returns whether a new position the same game as old position (with some
  // moves added). Returns false, if the position is completely different,
  // or if it's shorter than before.
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<Move>& moves);
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }
 private:
  void DeallocateTree();
  // A node which to start search from.
  Node* current_head_ = nullptr;
  // Root node of a game tree.
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/node.h
// begin /Users/syys/CLionProjects/lc0/src/utils/optionsdict.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
template <typename T>
class TypeDict {
 protected:
  struct V {
    const T& Get() const {
      is_used_ = true;
      return value_;
    }
    T& Get() {
      is_used_ = true;
      return value_;
    }
    void Set(const T& v) {
      is_used_ = false;
      value_ = v;
    }
    bool IsSet() const { return is_used_; }
   private:
    mutable bool is_used_ = false;
    T value_;
  };
  void EnsureNoUnusedOptions(const std::string& type_name,
                             const std::string& prefix) const {
    for (auto const& option : dict_) {
      if (!option.second.IsSet()) {
        throw Exception("Unknown " + type_name + " option: " + prefix +
                        option.first);
      }
    }
  }
  const std::unordered_map<std::string, V>& dict() const { return dict_; }
  std::unordered_map<std::string, V>* mutable_dict() { return &dict_; }
 private:
  std::unordered_map<std::string, V> dict_;
};
class OptionId {
 public:
  OptionId(const char* long_flag, const char* uci_option, const char* help_text,
           const char short_flag = '\0')
      : long_flag_(long_flag),
        uci_option_(uci_option),
        help_text_(help_text),
        short_flag_(short_flag) {}
  OptionId(const OptionId& other) = delete;
  bool operator==(const OptionId& other) const { return this == &other; }
  const char* long_flag() const { return long_flag_; }
  const char* uci_option() const { return uci_option_; }
  const char* help_text() const { return help_text_; }
  char short_flag() const { return short_flag_; }
 private:
  const char* const long_flag_;
  const char* const uci_option_;
  const char* const help_text_;
  const char short_flag_;
};
class OptionsDict : TypeDict<bool>,
                    TypeDict<int>,
                    TypeDict<std::string>,
                    TypeDict<float> {
 public:
  explicit OptionsDict(const OptionsDict* parent = nullptr)
      : parent_(parent), aliases_{this} {}
  // e.g. dict.Get<int>("threads")
  // Returns value of given type. Throws exception if not found.
  template <typename T>
  T Get(const std::string& key) const;
  template <typename T>
  T Get(const OptionId& option_id) const;
  // Returns the own value of given type (doesn't fall back to querying parent).
  // Returns nullopt if doesn't exist.
  template <typename T>
  std::optional<T> OwnGet(const std::string& key) const;
  template <typename T>
  std::optional<T> OwnGet(const OptionId& option_id) const;
  // Checks whether the given key exists for given type.
  template <typename T>
  bool Exists(const std::string& key) const;
  template <typename T>
  bool Exists(const OptionId& option_id) const;
  // Checks whether the given key exists for given type, and throws an exception
  // if not.
  template <typename T>
  void EnsureExists(const OptionId& option_id) const;
  // Checks whether the given key exists for given type. Does not fall back to
  // check parents.
  template <typename T>
  bool OwnExists(const std::string& key) const;
  template <typename T>
  bool OwnExists(const OptionId& option_id) const;
  // Returns value of given type. Returns default if not found.
  template <typename T>
  T GetOrDefault(const std::string& key, const T& default_val) const;
  template <typename T>
  T GetOrDefault(const OptionId& option_id, const T& default_val) const;
  // Sets value for a given type.
  template <typename T>
  void Set(const std::string& key, const T& value);
  template <typename T>
  void Set(const OptionId& option_id, const T& value);
  // Get reference to assign value to.
  template <typename T>
  T& GetRef(const std::string& key);
  template <typename T>
  T& GetRef(const OptionId& option_id);
  // Returns true when the value is not set anywhere maybe except the root
  // dictionary;
  template <typename T>
  bool IsDefault(const std::string& key) const;
  template <typename T>
  bool IsDefault(const OptionId& option_id) const;
  // Returns subdictionary. Throws exception if doesn't exist.
  const OptionsDict& GetSubdict(const std::string& name) const;
  // Returns subdictionary. Throws exception if doesn't exist.
  OptionsDict* GetMutableSubdict(const std::string& name);
  // Creates subdictionary. Throws exception if already exists.
  OptionsDict* AddSubdict(const std::string& name);
  // Returns list of subdictionaries.
  std::vector<std::string> ListSubdicts() const;
  // Adds alias dictionary.
  void AddAliasDict(const OptionsDict* dict);
  // Creates options dict from string. Example of a string:
  // option1=1, option_two = "string val", subdict(option3=3.14)
  //
  // the sub dictionary is containing a parent pointer refering
  // back to this object. You need to ensure, that this object
  // is still in scope, when the parent pointer is used
  void AddSubdictFromString(const std::string& str);
  // Throws an exception for the first option in the dict that has not been read
  // to find syntax errors in options added using AddSubdictFromString.
  void CheckAllOptionsRead(const std::string& path_from_parent) const;
  bool HasSubdict(const std::string& name) const;
 private:
  static std::string GetOptionId(const OptionId& option_id) {
    return std::to_string(reinterpret_cast<intptr_t>(&option_id));
  }
  const OptionsDict* parent_ = nullptr;
  std::map<std::string, OptionsDict> subdicts_;
  // Dictionaries where to search for "own" parameters. By default contains only
  // this.
  std::vector<const OptionsDict*> aliases_;
};
template <typename T>
T OptionsDict::Get(const std::string& key) const {
  for (const auto* alias : aliases_) {
    const auto value = alias->OwnGet<T>(key);
    if (value) return *value;
  }
  if (parent_) return parent_->Get<T>(key);
  throw Exception("Key [" + key + "] was not set in options.");
}
template <typename T>
T OptionsDict::Get(const OptionId& option_id) const {
  return Get<T>(GetOptionId(option_id));
}
template <typename T>
std::optional<T> OptionsDict::OwnGet(const std::string& key) const {
  const auto& dict = TypeDict<T>::dict();
  auto iter = dict.find(key);
  if (iter != dict.end()) {
    return iter->second.Get();
  }
  return {};
}
template <typename T>
std::optional<T> OptionsDict::OwnGet(const OptionId& option_id) const {
  return OwnGet<T>(GetOptionId(option_id));
}
template <typename T>
bool OptionsDict::Exists(const std::string& key) const {
  for (const auto* alias : aliases_) {
    if (alias->OwnExists<T>(key)) return true;
  }
  return parent_ && parent_->Exists<T>(key);
}
template <typename T>
bool OptionsDict::Exists(const OptionId& option_id) const {
  return Exists<T>(GetOptionId(option_id));
}
template <typename T>
void OptionsDict::EnsureExists(const OptionId& option_id) const {
  if (!OwnExists<T>(option_id)) {
    throw Exception(std::string("The flag --") + option_id.long_flag() +
                    " must be specified.");
  }
}
template <typename T>
bool OptionsDict::OwnExists(const std::string& key) const {
  const auto& dict = TypeDict<T>::dict();
  auto iter = dict.find(key);
  return iter != dict.end();
}
template <typename T>
bool OptionsDict::OwnExists(const OptionId& option_id) const {
  return OwnExists<T>(GetOptionId(option_id));
}
template <typename T>
T OptionsDict::GetOrDefault(const std::string& key,
                            const T& default_val) const {
  for (const auto* alias : aliases_) {
    const auto value = alias->OwnGet<T>(key);
    if (value) return *value;
  }
  if (parent_) return parent_->GetOrDefault<T>(key, default_val);
  return default_val;
}
template <typename T>
T OptionsDict::GetOrDefault(const OptionId& option_id,
                            const T& default_val) const {
  return GetOrDefault<T>(GetOptionId(option_id), default_val);
}
template <typename T>
void OptionsDict::Set(const std::string& key, const T& value) {
  (*TypeDict<T>::mutable_dict())[key].Set(value);
}
template <typename T>
void OptionsDict::Set(const OptionId& option_id, const T& value) {
  Set<T>(GetOptionId(option_id), value);
}
template <typename T>
T& OptionsDict::GetRef(const std::string& key) {
  return (*TypeDict<T>::mutable_dict())[key].Get();
}
template <typename T>
T& OptionsDict::GetRef(const OptionId& option_id) {
  return GetRef<T>(GetOptionId(option_id));
}
template <typename T>
bool OptionsDict::IsDefault(const std::string& key) const {
  if (!parent_) return true;
  for (const auto* alias : aliases_) {
    if (alias->OwnExists<T>(key)) return false;
  }
  return parent_->IsDefault<T>(key);
}
template <typename T>
bool OptionsDict::IsDefault(const OptionId& option_id) const {
  return IsDefault<T>(GetOptionId(option_id));
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/optionsdict.h
// begin /Users/syys/CLionProjects/lc0/src/utils/optionsparser.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class OptionsParser {
 public:
  OptionsParser();
  class Option {
   public:
    Option(const OptionId& id);
    virtual ~Option(){};
    // Set value from string.
    virtual void SetValue(const std::string& value, OptionsDict* dict) = 0;
   protected:
    const OptionId& GetId() const { return id_; }
    std::string GetUciOption() const { return id_.uci_option(); }
    std::string GetHelpText() const { return id_.help_text(); }
    std::string GetLongFlag() const { return id_.long_flag(); }
    char GetShortFlag() const { return id_.short_flag(); }
   private:
    virtual std::string GetOptionString(const OptionsDict& dict) const = 0;
    virtual bool ProcessLongFlag(const std::string& /*flag*/,
                                 const std::string& /*value*/,
                                 OptionsDict* /*dict*/) {
      return false;
    }
    virtual bool ProcessShortFlag(char /*flag*/, OptionsDict* /*dict*/) {
      return false;
    }
    virtual bool ProcessShortFlagWithValue(char /*flag*/,
                                           const std::string& /*value*/,
                                           OptionsDict* /*dict*/) {
      return false;
    }
    virtual std::string GetHelp(const OptionsDict& dict) const = 0;
    const OptionId& id_;
    bool hidden_ = false;
    friend class OptionsParser;
  };
  // Add an option to the list of available options (from command line flags
  // or UCI params)
  // Usage:
  // options->Add<StringOption>(name, func, long_flag, short_flag) = def_val;
  template <typename Option, typename... Args>
  typename Option::ValueType& Add(Args&&... args) {
    options_.emplace_back(
        std::make_unique<Option>(std::forward<Args>(args)...));
    return defaults_.GetRef<typename Option::ValueType>(
        options_.back()->GetId());
  }
  // Returns list of options in UCI format.
  std::vector<std::string> ListOptionsUci() const;
  // Set the UCI option from string value.
  void SetUciOption(const std::string& name, const std::string& value,
                    const std::string& context = "");
  // Hide this option from help and UCI.
  void HideOption(const OptionId& id);
  // Processes all flags from the command line and an optional
  // configuration file. Returns false if there is an invalid flag.
  bool ProcessAllFlags();
  // Processes either the command line or configuration file flags.
  bool ProcessFlags(const std::vector<std::string>& args);
  // Get the options dict for given context.
  const OptionsDict& GetOptionsDict(const std::string& context = {});
  // Gets the dictionary for given context which caller can modify.
  OptionsDict* GetMutableOptions(const std::string& context = {});
  // Gets the mutable list of default options.
  OptionsDict* GetMutableDefaultsOptions() { return &defaults_; }
  // Adds a subdictionary for a given context.
  void AddContext(const std::string&);
  // Prints help to std::cout.
  void ShowHelp() const;
 private:
  // Make all hidden options visible.
  void ShowHidden() const;
  // Returns an option based on the long flag.
  Option* FindOptionByLongFlag(const std::string& flag) const;
  // Returns an option based by its uci name.
  Option* FindOptionByUciName(const std::string& name) const;
  // Returns an option based by its id.
  Option* FindOptionById(const OptionId& id) const;
  std::vector<std::unique_ptr<Option>> options_;
  OptionsDict defaults_;
  OptionsDict& values_;
};
class StringOption : public OptionsParser::Option {
 public:
  using ValueType = std::string;
  StringOption(const OptionId& id);
  void SetValue(const std::string& value, OptionsDict* dict) override;
 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;
  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
};
class IntOption : public OptionsParser::Option {
 public:
  using ValueType = int;
  IntOption(const OptionId& id, int min, int max);
  void SetValue(const std::string& value, OptionsDict* dict) override;
 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;
  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
  int ValidateIntString(const std::string& val) const;
  int min_;
  int max_;
};
class FloatOption : public OptionsParser::Option {
 public:
  using ValueType = float;
  FloatOption(const OptionId& id, float min, float max);
  void SetValue(const std::string& value, OptionsDict* dict) override;
 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;
  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
  float min_;
  float max_;
};
class BoolOption : public OptionsParser::Option {
 public:
  using ValueType = bool;
  BoolOption(const OptionId& id);
  void SetValue(const std::string& value, OptionsDict* dict) override;
 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlag(char flag, OptionsDict* dict) override;
  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
  void ValidateBoolString(const std::string& val);
};
class ChoiceOption : public OptionsParser::Option {
 public:
  using ValueType = std::string;
  ChoiceOption(const OptionId& id, const std::vector<std::string>& choices);
  void SetValue(const std::string& value, OptionsDict* dict) override;
 private:
  std::string GetOptionString(const OptionsDict& dict) const override;
  bool ProcessLongFlag(const std::string& flag, const std::string& value,
                       OptionsDict* dict) override;
  std::string GetHelp(const OptionsDict& dict) const override;
  bool ProcessShortFlagWithValue(char flag, const std::string& value,
                                 OptionsDict* dict) override;
  ValueType GetVal(const OptionsDict&) const;
  void SetVal(OptionsDict* dict, const ValueType& val) const;
  std::vector<std::string> choices_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/optionsparser.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/params.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;
  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);
  // Parameter getters.
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  int GetMaxPrefetchBatch() const {
    return options_.Get<int>(kMaxPrefetchBatchId);
  }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const {
    return at_root ? kCpuctBaseAtRoot : kCpuctBase;
  }
  float GetCpuctFactor(bool at_root) const {
    return at_root ? kCpuctFactorAtRoot : kCpuctFactor;
  }
  bool GetTwoFoldDraws() const { return kTwoFoldDraws; }
  float GetTemperature() const { return options_.Get<float>(kTemperatureId); }
  float GetTemperatureVisitOffset() const {
    return options_.Get<float>(kTemperatureVisitOffsetId);
  }
  int GetTempDecayMoves() const { return options_.Get<int>(kTempDecayMovesId); }
  int GetTempDecayDelayMoves() const {
    return options_.Get<int>(kTempDecayDelayMovesId);
  }
  int GetTemperatureCutoffMove() const {
    return options_.Get<int>(kTemperatureCutoffMoveId);
  }
  float GetTemperatureEndgame() const {
    return options_.Get<float>(kTemperatureEndgameId);
  }
  float GetTemperatureWinpctCutoff() const {
    return options_.Get<float>(kTemperatureWinpctCutoffId);
  }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetNoiseAlpha() const { return kNoiseAlpha; }
  bool GetVerboseStats() const { return options_.Get<bool>(kVerboseStatsId); }
  bool GetLogLiveStats() const { return options_.Get<bool>(kLogLiveStatsId); }
  bool GetFpuAbsolute(bool at_root) const {
    return at_root ? kFpuAbsoluteAtRoot : kFpuAbsolute;
  }
  float GetFpuValue(bool at_root) const {
    return at_root ? kFpuValueAtRoot : kFpuValue;
  }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisits() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetStickyEndgames() const { return kStickyEndgames; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId); }
  bool GetPerPvCounters() const { return options_.Get<bool>(kPerPvCountersId); }
  std::string GetScoreType() const {
    return options_.Get<std::string>(kScoreTypeId);
  }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; }
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftSlope() const { return kMovesLeftSlope; }
  float GetMovesLeftConstantFactor() const { return kMovesLeftConstantFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const {
    return kMovesLeftQuadraticFactor;
  }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetSidetomoveDrawScore() const { return kDrawScoreSidetomove; }
  float GetOpponentDrawScore() const { return kDrawScoreOpponent; }
  float GetWhiteDrawDelta() const { return kDrawScoreWhite; }
  float GetBlackDrawDelta() const { return kDrawScoreBlack; }
  int GetMaxOutOfOrderEvals() const { return kMaxOutOfOrderEvals; }
  float GetNpsLimit() const { return kNpsLimit; }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; }
  int GetTaskWorkersPerSearchWorker() const {
    return kTaskWorkersPerSearchWorker;
  }
  int GetMinimumWorkSizeForProcessing() const {
    return kMinimumWorkSizeForProcessing;
  }
  int GetMinimumWorkSizeForPicking() const {
    return kMinimumWorkSizeForPicking;
  }
  int GetMinimumRemainingWorkSizeForPicking() const {
    return kMinimumRemainingWorkSizeForPicking;
  }
  int GetMinimumWorkPerTaskForProcessing() const {
    return kMinimumWorkPerTaskForProcessing;
  }
  int GetIdlingMinimumWork() const { return kIdlingMinimumWork; }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  int GetMaxCollisionVisitsScalingStart() const {
    return kMaxCollisionVisitsScalingStart;
  }
  int GetMaxCollisionVisitsScalingEnd() const {
    return kMaxCollisionVisitsScalingEnd;
  }
  float GetMaxCollisionVisitsScalingPower() const {
    return kMaxCollisionVisitsScalingPower;
  }
  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kTwoFoldDrawsId;
  static const OptionId kTemperatureId;
  static const OptionId kTempDecayMovesId;
  static const OptionId kTempDecayDelayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kNoiseAlphaId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  static const OptionId kPolicySoftmaxTempId;
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kStickyEndgamesId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kPerPvCountersId;
  static const OptionId kScoreTypeId;
  static const OptionId kHistoryFillId;
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftConstantFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kMovesLeftSlopeId;
  static const OptionId kDisplayCacheUsageId;
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreSidetomoveId;
  static const OptionId kDrawScoreOpponentId;
  static const OptionId kDrawScoreWhiteId;
  static const OptionId kDrawScoreBlackId;
  static const OptionId kMaxOutOfOrderEvalsId;
  static const OptionId kNpsLimitId;
  static const OptionId kSolidTreeThresholdId;
  static const OptionId kTaskWorkersPerSearchWorkerId;
  static const OptionId kMinimumWorkSizeForProcessingId;
  static const OptionId kMinimumWorkSizeForPickingId;
  static const OptionId kMinimumRemainingWorkSizeForPickingId;
  static const OptionId kMinimumWorkPerTaskForProcessingId;
  static const OptionId kIdlingMinimumWorkId;
  static const OptionId kThreadIdlingThresholdId;
  static const OptionId kMaxCollisionVisitsScalingStartId;
  static const OptionId kMaxCollisionVisitsScalingEndId;
  static const OptionId kMaxCollisionVisitsScalingPowerId;
 private:
  const OptionsDict& options_;
  // Cached parameter values. Values have to be cached if either:
  // 1. Parameter is accessed often and has to be cached for performance
  // reasons.
  // 2. Parameter has to stay the say during the search.
  // TODO(crem) Some of those parameters can be converted to be dynamic after
  //            trivial search optimizations.
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const bool kTwoFoldDraws;
  const float kNoiseEpsilon;
  const float kNoiseAlpha;
  const bool kFpuAbsolute;
  const float kFpuValue;
  const bool kFpuAbsoluteAtRoot;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp;
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kStickyEndgames;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill;
  const int kMiniBatchSize;
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftSlope;
  const float kMovesLeftConstantFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const bool kDisplayCacheUsage;
  const int kMaxConcurrentSearchers;
  const float kDrawScoreSidetomove;
  const float kDrawScoreOpponent;
  const float kDrawScoreWhite;
  const float kDrawScoreBlack;
  const int kMaxOutOfOrderEvals;
  const float kNpsLimit;
  const int kSolidTreeThreshold;
  const int kTaskWorkersPerSearchWorker;
  const int kMinimumWorkSizeForProcessing;
  const int kMinimumWorkSizeForPicking;
  const int kMinimumRemainingWorkSizeForPicking;
  const int kMinimumWorkPerTaskForProcessing;
  const int kIdlingMinimumWork;
  const int kThreadIdlingThreshold;
  const int kMaxCollisionVisitsScalingStart;
  const int kMaxCollisionVisitsScalingEnd;
  const float kMaxCollisionVisitsScalingPower;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/params.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/stoppers/timemgr.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Various statistics that search sends to stoppers for their stopping decision.
// It is expected that this structure will grow.
struct IterationStats {
  int64_t time_since_movestart = 0;
  int64_t time_since_first_batch = 0;
  int64_t total_nodes = 0;
  int64_t nodes_since_movestart = 0;
  int64_t batches_since_movestart = 0;
  int average_depth = 0;
  std::vector<uint32_t> edge_n;
  // TODO: remove this in favor of time_usage_hint_=kImmediateMove when
  // smooth time manager is the default.
  bool win_found = false;
  int num_losing_edges = 0;
  enum class TimeUsageHint { kNormal, kNeedMoreTime, kImmediateMove };
  TimeUsageHint time_usage_hint_ = TimeUsageHint::kNormal;
};
// Hints from stoppers back to the search engine. Currently include:
// 1. EstimatedRemainingTime -- for search watchdog thread to know when to
// expect running out of time.
// 2. EstimatedPlayouts -- for smart pruning at root (not pick root nodes that
// cannot potentially become good).
class StoppersHints {
 public:
  StoppersHints();
  void Reset();
  void UpdateEstimatedRemainingTimeMs(int64_t v);
  int64_t GetEstimatedRemainingTimeMs() const;
  void UpdateEstimatedRemainingPlayouts(int64_t v);
  int64_t GetEstimatedRemainingPlayouts() const;
  void UpdateEstimatedNps(float v);
  std::optional<float> GetEstimatedNps() const;
 private:
  int64_t remaining_time_ms_;
  int64_t remaining_playouts_;
  std::optional<float> estimated_nps_;
};
// Interface for search stopper.
// Note that:
// 1. Stoppers are shared between all search threads, so if stopper has mutable
// varibles, it has to think about concurrency (mutex/atomics)
// (maybe in future it will be changed).
// 2. IterationStats and StoppersHints are per search thread, so access to
// them is fine without synchronization.
// 3. OnSearchDone is guaranteed to be called once (i.e. from only one thread).
class SearchStopper {
 public:
  virtual ~SearchStopper() = default;
  // Question to a stopper whether search should stop.
  // Search statistics is sent via IterationStats, the stopper can optionally
  // send hints to the search through StoppersHints.
  virtual bool ShouldStop(const IterationStats&, StoppersHints*) = 0;
  // Is called when search is done.
  virtual void OnSearchDone(const IterationStats&) {}
};
class TimeManager {
 public:
  virtual ~TimeManager() = default;
  virtual std::unique_ptr<SearchStopper> GetStopper(const GoParams& params,
                                                    const NodeTree& tree) = 0;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/timemgr.h
// begin /Users/syys/CLionProjects/lc0/src/syzygy/syzygy.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
enum WDLScore {
  WDL_LOSS = -2,          // Loss
  WDL_BLESSED_LOSS = -1,  // Loss, but draw under 50-move rule
  WDL_DRAW = 0,           // Draw
  WDL_CURSED_WIN = 1,     // Win, but draw under 50-move rule
  WDL_WIN = 2,            // Win
};
// Possible states after a probing operation
enum ProbeState {
  FAIL = 0,              // Probe failed (missing file table)
  OK = 1,                // Probe succesful
  CHANGE_STM = -1,       // DTZ should check the other side
  ZEROING_BEST_MOVE = 2  // Best move zeroes DTZ (capture or pawn move)
};
class SyzygyTablebaseImpl;
// Provides methods to load and probe syzygy tablebases.
// Thread safe methods are thread safe subject to the non-thread sfaety
// conditions of the init method.
class SyzygyTablebase {
 public:
  SyzygyTablebase();
  virtual ~SyzygyTablebase();
  // Current maximum number of pieces on board that can be probed for. Will
  // be 0 unless initialized with tablebase paths.
  // Thread safe.
  int max_cardinality() { return max_cardinality_; }
  // Allows for the tablebases being used to be changed. This method is not
  // thread safe, there must be no concurrent usage while this method is
  // running. All other thread safe method calls must be strictly ordered with
  // respect to this method.
  bool init(const std::string& paths);
  // Probes WDL tables for the given position to determine a WDLScore.
  // Thread safe.
  // Result is only strictly valid for positions with 0 ply 50 move counter.
  // Probe state will return FAIL if the position is not in the tablebase.
  WDLScore probe_wdl(const Position& pos, ProbeState* result);
  // Probes DTZ tables for the given position to determine the number of ply
  // before a zeroing move under optimal play.
  // Thread safe.
  // Probe state will return FAIL if the position is not in the tablebase.
  int probe_dtz(const Position& pos, ProbeState* result);
  // Probes DTZ tables to determine which moves are on the optimal play path.
  // Assumes the position is one reached such that the side to move has been
  // performing optimal play moves since the last 50 move counter reset.
  // has_repeated should be whether there are any repeats since last 50 move
  // counter reset.
  // Thread safe.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe(const Position& pos, bool has_repeated,
                  std::vector<Move>* safe_moves);
  // Probes WDL tables to determine which moves might be on the optimal play
  // path. If 50 move ply counter is non-zero some (or maybe even all) of the
  // returned safe moves in a 'winning' position, may actually be draws.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe_wdl(const Position& pos, std::vector<Move>* safe_moves);
 private:
  template <bool CheckZeroingMoves = false>
  WDLScore search(const Position& pos, ProbeState* result);
  std::string paths_;
  // Caches the max_cardinality from the impl, as max_cardinality may be a hot
  // path.
  int max_cardinality_;
  std::unique_ptr<SyzygyTablebaseImpl> impl_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/syzygy/syzygy.h
// begin /Users/syys/CLionProjects/lc0/src/utils/numa.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class Numa {
 public:
  Numa() = delete;
  // Initialize and display statistics about processor configuration.
  static void Init();
  // Bind thread to processor group.
  static void BindThread(int id);
 private:
  static int threads_per_core_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/numa.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/search.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class Search {
 public:
  Search(const NodeTree& tree, Network* network,
         std::unique_ptr<UciResponder> uci_responder,
         const MoveList& searchmoves,
         std::chrono::steady_clock::time_point start_time,
         std::unique_ptr<SearchStopper> stopper, bool infinite,
         const OptionsDict& options, NNCache* cache,
         SyzygyTablebase* syzygy_tb);
  ~Search();
  // Starts worker threads and returns immediately.
  void StartThreads(size_t how_many);
  // Starts search with k threads and wait until it finishes.
  void RunBlocking(size_t threads);
  // Stops search. At the end bestmove will be returned. The function is not
  // blocking, so it returns before search is actually done.
  void Stop();
  // Stops search, but does not return bestmove. The function is not blocking.
  void Abort();
  // Blocks until all worker thread finish.
  void Wait();
  // Returns whether search is active. Workers check that to see whether another
  // search iteration is needed.
  bool IsSearchActive() const;
  // Returns best move, from the point of view of white player. And also ponder.
  // May or may not use temperature, according to the settings.
  std::pair<Move, Move> GetBestMove();
  // Returns the evaluation of the best move, WITHOUT temperature. This differs
  // from the above function; with temperature enabled, these two functions may
  // return results from different possible moves. If @move and @is_terminal are
  // not nullptr they are set to the best move and whether it leads to a
  // terminal node respectively.
  Eval GetBestEval(Move* move = nullptr, bool* is_terminal = nullptr) const;
  // Returns the total number of playouts in the search.
  std::int64_t GetTotalPlayouts() const;
  // Returns the search parameters.
  const SearchParams& GetParams() const { return params_; }
  // If called after GetBestMove, another call to GetBestMove will have results
  // from temperature having been applied again.
  void ResetBestMove();
  // Returns NN eval for a given node from cache, if that node is cached.
  NNCacheLock GetCachedNNEval(const Node* node) const;
 private:
  // Computes the best move, maybe with temperature (according to the settings).
  void EnsureBestMoveKnown();
  // Returns a child with most visits, with or without temperature.
  // NoTemperature is safe to use on non-extended nodes, while WithTemperature
  // accepts only nodes with at least 1 visited child.
  EdgeAndNode GetBestChildNoTemperature(Node* parent, int depth) const;
  std::vector<EdgeAndNode> GetBestChildrenNoTemperature(Node* parent, int count,
                                                        int depth) const;
  EdgeAndNode GetBestRootChildWithTemperature(float temperature) const;
  int64_t GetTimeSinceStart() const;
  int64_t GetTimeSinceFirstBatch() const;
  void MaybeTriggerStop(const IterationStats& stats, StoppersHints* hints);
  void MaybeOutputInfo();
  void SendUciInfo();  // Requires nodes_mutex_ to be held.
  // Sets stop to true and notifies watchdog thread.
  void FireStopInternal();
  void SendMovesStats() const;
  // Function which runs in a separate thread and watches for time and
  // uci `stop` command;
  void WatchdogThread();
  // Fills IterationStats with global (rather than per-thread) portion of search
  // statistics. Currently all stats there (in IterationStats) are global
  // though.
  void PopulateCommonIterationStats(IterationStats* stats);
  // Returns verbose information about given node, as vector of strings.
  // Node can only be root or ponder (depth 1).
  std::vector<std::string> GetVerboseStats(Node* node) const;
  // Returns the draw score at the root of the search. At odd depth pass true to
  // the value of @is_odd_depth to change the sign of the draw score.
  // Depth of a root node is 0 (even number).
  float GetDrawScore(bool is_odd_depth) const;
  // Ensure that all shared collisions are cancelled and clear them out.
  void CancelSharedCollisions();
  mutable Mutex counters_mutex_ ACQUIRED_AFTER(nodes_mutex_);
  // Tells all threads to stop.
  std::atomic<bool> stop_{false};
  // Condition variable used to watch stop_ variable.
  std::condition_variable watchdog_cv_;
  // Tells whether it's ok to respond bestmove when limits are reached.
  // If false (e.g. during ponder or `go infinite`) the search stops but nothing
  // is responded until `stop` uci command.
  bool ok_to_respond_bestmove_ GUARDED_BY(counters_mutex_) = true;
  // There is already one thread that responded bestmove, other threads
  // should not do that.
  bool bestmove_is_sent_ GUARDED_BY(counters_mutex_) = false;
  // Stored so that in the case of non-zero temperature GetBestMove() returns
  // consistent results.
  Move final_bestmove_ GUARDED_BY(counters_mutex_);
  Move final_pondermove_ GUARDED_BY(counters_mutex_);
  std::unique_ptr<SearchStopper> stopper_ GUARDED_BY(counters_mutex_);
  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);
  Node* root_node_;
  NNCache* cache_;
  SyzygyTablebase* syzygy_tb_;
  // Fixed positions which happened before the search.
  const PositionHistory& played_history_;
  Network* const network_;
  const SearchParams params_;
  const MoveList searchmoves_;
  const std::chrono::steady_clock::time_point start_time_;
  int64_t initial_visits_;
  // root_is_in_dtz_ must be initialized before root_move_filter_.
  bool root_is_in_dtz_ = false;
  // tb_hits_ must be initialized before root_move_filter_.
  std::atomic<int> tb_hits_{0};
  const MoveList root_move_filter_;
  mutable SharedMutex nodes_mutex_;
  EdgeAndNode current_best_edge_ GUARDED_BY(nodes_mutex_);
  Edge* last_outputted_info_edge_ GUARDED_BY(nodes_mutex_) = nullptr;
  ThinkingInfo last_outputted_uci_info_ GUARDED_BY(nodes_mutex_);
  int64_t total_playouts_ GUARDED_BY(nodes_mutex_) = 0;
  int64_t total_batches_ GUARDED_BY(nodes_mutex_) = 0;
  // Maximum search depth = length of longest path taken in PickNodetoExtend.
  uint16_t max_depth_ GUARDED_BY(nodes_mutex_) = 0;
  // Cumulative depth of all paths taken in PickNodetoExtend.
  uint64_t cum_depth_ GUARDED_BY(nodes_mutex_) = 0;
  std::optional<std::chrono::steady_clock::time_point> nps_start_time_
      GUARDED_BY(counters_mutex_);
  std::atomic<int> pending_searchers_{0};
  std::atomic<int> backend_waiting_counter_{0};
  std::atomic<int> thread_count_{0};
  std::vector<std::pair<Node*, int>> shared_collisions_
      GUARDED_BY(nodes_mutex_);
  std::unique_ptr<UciResponder> uci_responder_;
  friend class SearchWorker;
};
// Single thread worker of the search engine.
// That used to be just a function Search::Worker(), but to parallelize it
// within one thread, have to split into stages.
class SearchWorker {
 public:
  SearchWorker(Search* search, const SearchParams& params, int id)
      : search_(search),
        history_(search_->played_history_),
        params_(params),
        moves_left_support_(search_->network_->GetCapabilities().moves_left !=
                            pblczero::NetworkFormat::MOVES_LEFT_NONE) {
    Numa::BindThread(id);
    for (int i = 0; i < params.GetTaskWorkersPerSearchWorker(); i++) {
      task_workspaces_.emplace_back();
      task_threads_.emplace_back([this, i]() {
        Numa::BindThread(i);
        this->RunTasks(i);
      });
    }
  }
  ~SearchWorker() {
    {
      task_count_.store(-1, std::memory_order_release);
      Mutex::Lock lock(picking_tasks_mutex_);
      exiting_ = true;
      task_added_.notify_all();
    }
    for (size_t i = 0; i < task_threads_.size(); i++) {
      task_threads_[i].join();
    }
  }
  // Runs iterations while needed.
  void RunBlocking() {
    LOGFILE << "Started search thread.";
    try {
      // A very early stop may arrive before this point, so the test is at the
      // end to ensure at least one iteration runs before exiting.
      do {
        ExecuteOneIteration();
      } while (search_->IsSearchActive());
    } catch (std::exception& e) {
      std::cerr << "Unhandled exception in worker thread: " << e.what()
                << std::endl;
      abort();
    }
  }
  // Does one full iteration of MCTS search:
  // 1. Initialize internal structures.
  // 2. Gather minibatch.
  // 3. Prefetch into cache.
  // 4. Run NN computation.
  // 5. Retrieve NN computations (and terminal values) into nodes.
  // 6. Propagate the new nodes' information to all their parents in the tree.
  // 7. Update the Search's status and progress information.
  void ExecuteOneIteration();
  // The same operations one by one:
  // 1. Initialize internal structures.
  // @computation is the computation to use on this iteration.
  void InitializeIteration(std::unique_ptr<NetworkComputation> computation);
  // 2. Gather minibatch.
  void GatherMinibatch();
  // Variant for multigather path.
  void GatherMinibatch2();
  // 2b. Copy collisions into shared_collisions_.
  void CollectCollisions();
  // 3. Prefetch into cache.
  void MaybePrefetchIntoCache();
  // 4. Run NN computation.
  void RunNNComputation();
  // 5. Retrieve NN computations (and terminal values) into nodes.
  void FetchMinibatchResults();
  // 6. Propagate the new nodes' information to all their parents in the tree.
  void DoBackupUpdate();
  // 7. Update the Search's status and progress information.
  void UpdateCounters();
 private:
  struct NodeToProcess {
    bool IsExtendable() const { return !is_collision && !node->IsTerminal(); }
    bool IsCollision() const { return is_collision; }
    bool CanEvalOutOfOrder() const {
      return is_cache_hit || node->IsTerminal();
    }
    // The node to extend.
    Node* node;
    // Value from NN's value head, or -1/0/1 for terminal nodes.
    float v;
    // Draw probability for NN's with WDL value head.
    float d;
    // Estimated remaining plies left.
    float m;
    int multivisit = 0;
    // If greater than multivisit, and other parameters don't imply a lower
    // limit, multivist could be increased to this value without additional
    // change in outcome of next selection.
    int maxvisit = 0;
    uint16_t depth;
    bool nn_queried = false;
    bool is_cache_hit = false;
    bool is_collision = false;
    int probability_transform = 0;
    // Details only populated in the multigather path.
    // Only populated for visits,
    std::vector<Move> moves_to_visit;
    // Details that are filled in as we go.
    uint64_t hash;
    NNCacheLock lock;
    std::vector<uint16_t> probabilities_to_cache;
    InputPlanes input_planes;
    mutable int last_idx = 0;
    bool ooo_completed = false;
    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count) {
      return NodeToProcess(node, depth, true, collision_count, 0);
    }
    static NodeToProcess Collision(Node* node, uint16_t depth,
                                   int collision_count, int max_count) {
      return NodeToProcess(node, depth, true, collision_count, max_count);
    }
    static NodeToProcess Visit(Node* node, uint16_t depth) {
      return NodeToProcess(node, depth, false, 1, 0);
    }
    // Methods to allow NodeToProcess to conform as a 'Computation'. Only safe
    // to call if is_cache_hit is true in the multigather path.
    float GetQVal(int) const { return lock->q; }
    float GetDVal(int) const { return lock->d; }
    float GetMVal(int) const { return lock->m; }
    float GetPVal(int, int move_id) const {
      const auto& moves = lock->p;
      int total_count = 0;
      while (total_count < moves.size()) {
        // Optimization: usually moves are stored in the same order as queried.
        const auto& move = moves[last_idx++];
        if (last_idx == moves.size()) last_idx = 0;
        if (move.first == move_id) return move.second;
        ++total_count;
      }
      assert(false);  // Move not found.
      return 0;
    }
   private:
    NodeToProcess(Node* node, uint16_t depth, bool is_collision, int multivisit,
                  int max_count)
        : node(node),
          multivisit(multivisit),
          maxvisit(max_count),
          depth(depth),
          is_collision(is_collision) {}
  };
  // Holds per task worker scratch data
  struct TaskWorkspace {
    std::array<Node::Iterator, 256> cur_iters;
    std::vector<std::unique_ptr<std::array<int, 256>>> vtp_buffer;
    std::vector<std::unique_ptr<std::array<int, 256>>> visits_to_perform;
    std::vector<int> vtp_last_filled;
    std::vector<int> current_path;
    std::vector<Move> moves_to_path;
    PositionHistory history;
    TaskWorkspace() {
      vtp_buffer.reserve(30);
      visits_to_perform.reserve(30);
      vtp_last_filled.reserve(30);
      current_path.reserve(30);
      moves_to_path.reserve(30);
      history.Reserve(30);
    }
  };
  struct PickTask {
    enum PickTaskType { kGathering, kProcessing };
    PickTaskType task_type;
    // For task type gathering.
    Node* start;
    int base_depth;
    int collision_limit;
    std::vector<Move> moves_to_base;
    std::vector<NodeToProcess> results;
    // Task type post gather processing.
    int start_idx;
    int end_idx;
    bool complete = false;
    PickTask(Node* node, uint16_t depth, const std::vector<Move>& base_moves,
             int collision_limit)
        : task_type(kGathering),
          start(node),
          base_depth(depth),
          collision_limit(collision_limit),
          moves_to_base(base_moves) {}
    PickTask(int start_idx, int end_idx)
        : task_type(kProcessing), start_idx(start_idx), end_idx(end_idx) {}
  };
  NodeToProcess PickNodeToExtend(int collision_limit);
  void ExtendNode(Node* node, int depth);
  bool AddNodeToComputation(Node* node, bool add_if_cached, int* transform_out);
  int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth);
  void DoBackupUpdateSingleNode(const NodeToProcess& node_to_process);
  // Returns whether a node's bounds were set based on its children.
  bool MaybeSetBounds(Node* p, float m, int* n_to_fix, float* v_delta,
                      float* d_delta, float* m_delta) const;
  void PickNodesToExtend(int collision_limit);
  void PickNodesToExtendTask(Node* starting_point, int collision_limit,
                             int base_depth,
                             const std::vector<Move>& moves_to_base,
                             std::vector<NodeToProcess>* receiver,
                             TaskWorkspace* workspace);
  void EnsureNodeTwoFoldCorrectForDepth(Node* node, int depth);
  void ProcessPickedTask(int batch_start, int batch_end,
                         TaskWorkspace* workspace);
  void ExtendNode(Node* node, int depth, const std::vector<Move>& moves_to_add,
                  PositionHistory* history);
  template <typename Computation>
  void FetchSingleNodeResult(NodeToProcess* node_to_process,
                             const Computation& computation,
                             int idx_in_computation);
  void RunTasks(int tid);
  void ResetTasks();
  // Returns how many tasks there were.
  int WaitForTasks();
  Search* const search_;
  // List of nodes to process.
  std::vector<NodeToProcess> minibatch_;
  std::unique_ptr<CachingComputation> computation_;
  // History is reset and extended by PickNodeToExtend().
  PositionHistory history_;
  int number_out_of_order_ = 0;
  const SearchParams& params_;
  std::unique_ptr<Node> precached_node_;
  const bool moves_left_support_;
  IterationStats iteration_stats_;
  StoppersHints latest_time_manager_hints_;
  // Multigather task related fields.
  Mutex picking_tasks_mutex_;
  std::vector<PickTask> picking_tasks_;
  std::atomic<int> task_count_ = -1;
  std::atomic<int> task_taking_started_ = 0;
  std::atomic<int> tasks_taken_ = 0;
  std::atomic<int> completed_tasks_ = 0;
  std::condition_variable task_added_;
  std::vector<std::thread> task_threads_;
  std::vector<TaskWorkspace> task_workspaces_;
  TaskWorkspace main_workspace_;
  bool exiting_ = false;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/search.h
// begin /Users/syys/CLionProjects/lc0/src/neural/loader.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
using FloatVector = std::vector<float>;
using FloatVectors = std::vector<FloatVector>;
using WeightsFile = pblczero::Net;
// Read weights file and fill the weights structure.
WeightsFile LoadWeightsFromFile(const std::string& filename);
// Tries to find a file which looks like a weights file, and located in
// directory of binary_name or one of subdirectories. If there are several such
// files, returns one which has the latest modification date.
std::string DiscoverWeightsFile();
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/loader.h
// begin /Users/syys/CLionProjects/lc0/src/neural/factory.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class NetworkFactory {
 public:
  using FactoryFunc = std::function<std::unique_ptr<Network>(
      const std::optional<WeightsFile>&, const OptionsDict&)>;
  static NetworkFactory* Get();
  // Registers network so it can be created by name.
  // @name -- name
  // @options -- options to pass to the network
  // @priority -- how high should be the network in the list. The network with
  //              the highest priority is the default.
  class Register {
   public:
    Register(const std::string& name, FactoryFunc factory, int priority = 0);
  };
  // Add the network/backend parameters to the options dictionary.
  static void PopulateOptions(OptionsParser* options);
  // Returns list of backend names, sorted by priority (higher priority first).
  std::vector<std::string> GetBackendsList() const;
  // Creates a backend given name and config.
  std::unique_ptr<Network> Create(const std::string& network,
                                  const std::optional<WeightsFile>&,
                                  const OptionsDict& options);
  // Helper function to load the network from the options. Returns nullptr
  // if no network options changed since the previous call.
  static std::unique_ptr<Network> LoadNetwork(const OptionsDict& options);
  // Parameter IDs.
  static const OptionId kWeightsId;
  static const OptionId kBackendId;
  static const OptionId kBackendOptionsId;
  struct BackendConfiguration {
    BackendConfiguration() = default;
    BackendConfiguration(const OptionsDict& options);
    std::string weights_path;
    std::string backend;
    std::string backend_options;
    bool operator==(const BackendConfiguration& other) const;
    bool operator!=(const BackendConfiguration& other) const {
      return !operator==(other);
    }
    bool operator<(const BackendConfiguration& other) const {
      return std::tie(weights_path, backend, backend_options) <
             std::tie(other.weights_path, other.backend, other.backend_options);
    }
  };
 private:
  void RegisterNetwork(const std::string& name, FactoryFunc factory,
                       int priority);
  NetworkFactory() {}
  struct Factory {
    Factory(const std::string& name, FactoryFunc factory, int priority)
        : name(name), factory(factory), priority(priority) {}
    bool operator<(const Factory& other) const {
      if (priority != other.priority) return priority > other.priority;
      return name < other.name;
    }
    std::string name;
    FactoryFunc factory;
    int priority;
  };
  std::vector<Factory> factories_;
  friend class Register;
};
#define REGISTER_NETWORK_WITH_COUNTER2(name, func, priority, counter) \
  namespace {                                                         \
  static NetworkFactory::Register regH38fhs##counter(                 \
      name,                                                           \
      [](const std::optional<WeightsFile>& w, const OptionsDict& o) { \
        return func(w, o);                                            \
      },                                                              \
      priority);                                                      \
  }
#define REGISTER_NETWORK_WITH_COUNTER(name, func, priority, counter) \
  REGISTER_NETWORK_WITH_COUNTER2(name, func, priority, counter)
// Registers a Network.
// Constructor of a network class must have parameters:
// (const Weights& w, const OptionsDict& o)
// @name -- name under which the backend will be known in configs.
// @func -- Factory function for a backend.
//          std::unique_ptr<Network>(const WeightsFile&, const OptionsDict&)
// @priority -- numeric priority of a backend. Higher is higher, highest number
// is the default backend.
#define REGISTER_NETWORK(name, func, priority) \
  REGISTER_NETWORK_WITH_COUNTER(name, func, priority, __LINE__)
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/factory.h
// begin /Users/syys/CLionProjects/lc0/src/benchmark/benchmark.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class Benchmark{
 public:
  Benchmark() = default;
  // Same positions as Stockfish uses.
  std::vector<std::string> positions = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
      "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
      "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14 moves "
      "d4e6",
      "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14 moves "
      "g2g4",
      "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
      "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
      "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
      "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
      "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
      "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
      "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
      "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
      "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
      "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
      "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
      "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
      "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1 moves g5g6 f3e3 g6g5 e3f3",
      "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
      "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
      "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
      "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
      "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
      "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
      "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
      "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
      "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
      "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
      "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
      "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
      "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
      "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
      "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40"
  };
  void Run();
  void OnBestMove(const BestMoveInfo& move);
  void OnInfo(const std::vector<ThinkingInfo>& infos);
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/benchmark/benchmark.h
// begin /Users/syys/CLionProjects/lc0/src/engine.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
struct CurrentPosition {
  std::string fen;
  std::vector<std::string> moves;
};
class EngineController {
 public:
  EngineController(std::unique_ptr<UciResponder> uci_responder,
                   const OptionsDict& options);
  ~EngineController() {
    // Make sure search is destructed first, and it still may be running in
    // a separate thread.
    search_.reset();
  }
  void PopulateOptions(OptionsParser* options);
  // Blocks.
  void EnsureReady();
  // Must not block.
  void NewGame();
  // Blocks.
  void SetPosition(const std::string& fen,
                   const std::vector<std::string>& moves);
  // Must not block.
  void Go(const GoParams& params);
  void PonderHit();
  // Must not block.
  void Stop();
  Position ApplyPositionMoves();
 private:
  void UpdateFromUciOptions();
  void SetupPosition(const std::string& fen,
                     const std::vector<std::string>& moves);
  void ResetMoveTimer();
  void CreateFreshTimeManager();
  const OptionsDict& options_;
  std::unique_ptr<UciResponder> uci_responder_;
  // Locked means that there is some work to wait before responding readyok.
  RpSharedMutex busy_mutex_;
  using SharedLock = std::shared_lock<RpSharedMutex>;
  std::unique_ptr<TimeManager> time_manager_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<NodeTree> tree_;
  std::unique_ptr<SyzygyTablebase> syzygy_tb_;
  std::unique_ptr<Network> network_;
  NNCache cache_;
  // Store current TB and network settings to track when they change so that
  // they are reloaded.
  std::string tb_paths_;
  NetworkFactory::BackendConfiguration network_configuration_;
  // The current position as given with SetPosition. For normal (ie. non-ponder)
  // search, the tree is set up with this position, however, during ponder we
  // actually search the position one move earlier.
  CurrentPosition current_position_;
  GoParams go_params_;
  std::optional<std::chrono::steady_clock::time_point> move_start_time_;
  // If true we can reset move_start_time_ in "Go".
  bool strict_uci_timing_;
};
class EngineLoop : public UciLoop {
 public:
  EngineLoop();
  void RunLoop() override;
  void CmdUci() override;
  void CmdIsReady() override;
  void CmdSetOption(const std::string& name, const std::string& value,
                    const std::string& context) override;
  void CmdUciNewGame() override;
  void CmdPosition(const std::string& position,
                   const std::vector<std::string>& moves) override;
  void CmdFen() override;
  void CmdGo(const GoParams& params) override;
  void CmdPonderHit() override;
  void CmdStop() override;
 private:
  OptionsParser options_;
  EngineController engine_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/engine.h
// begin /Users/syys/CLionProjects/lc0/src/lc0ctl/describenet.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
void DescribeNetworkCmd();
void ShowNetworkGenericInfo(const pblczero::Net& weights);
void ShowNetworkFormatInfo(const pblczero::Net& weights);
void ShowNetworkTrainingInfo(const pblczero::Net& weights);
void ShowNetworkWeightsInfo(const pblczero::Net& weights);
void ShowNetworkOnnxInfo(const pblczero::Net& weights,
                         bool show_onnx_internals);
void ShowAllNetworkInfo(const pblczero::Net& weights);
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/lc0ctl/describenet.h
// begin /Users/syys/CLionProjects/lc0/src/lc0ctl/leela2onnx.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
void ConvertLeelaToOnnx();
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/lc0ctl/leela2onnx.h
// begin /Users/syys/CLionProjects/lc0/src/lc0ctl/onnx2leela.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
void ConvertOnnxToLeela();
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/lc0ctl/onnx2leela.h
// begin /Users/syys/CLionProjects/lc0/src/chess/pgn.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
struct Opening {
  std::string start_fen = ChessBoard::kStartposFen;
  MoveList moves;
};
inline bool GzGetLine(gzFile file, std::string& line) {
  bool flag = false;
  char s[2000];
  line.clear();
  while (gzgets(file, s, sizeof(s))) {
    flag = true;
    line += s;
    auto r = line.find_last_of('\n');
    if (r != std::string::npos) {
      line.erase(r);
      break;
    }
  }
  return flag;
}
class PgnReader {
 public:
  void AddPgnFile(const std::string& filepath) {
    const gzFile file = gzopen(filepath.c_str(), "r");
    if (!file) {
      throw Exception(errno == ENOENT ? "Opening book file not found."
                                      : "Error opening opening book file.");
    }
    std::string line;
    bool in_comment = false;
    bool started = false;
    while (GzGetLine(file, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      // TODO: support line breaks in tags to ensure they are properly ignored.
      if (line.empty() || line[0] == '[') {
        if (started) {
          Flush();
          started = false;
        }
        auto uc_line = line;
        std::transform(
            uc_line.begin(), uc_line.end(), uc_line.begin(),
            [](unsigned char c) { return std::toupper(c); }  // correct
        );
        if (uc_line.find("[FEN \"", 0) == 0) {
          auto start_trimmed = line.substr(6);
          cur_startpos_ = start_trimmed.substr(0, start_trimmed.find('"'));
          cur_board_.SetFromFen(cur_startpos_);
        }
        continue;
      }
      // Must have at least one non-tag non-empty line in order to be considered
      // a game.
      started = true;
      // Handle braced comments.
      int cur_offset = 0;
      while ((in_comment && line.find('}', cur_offset) != std::string::npos) ||
             (!in_comment && line.find('{', cur_offset) != std::string::npos)) {
        if (in_comment && line.find('}', cur_offset) != std::string::npos) {
          line = line.substr(0, cur_offset) +
                 line.substr(line.find('}', cur_offset) + 1);
          in_comment = false;
        } else {
          cur_offset = line.find('{', cur_offset);
          in_comment = true;
        }
      }
      if (in_comment) {
        line = line.substr(0, cur_offset);
      }
      // Trim trailing comment.
      if (line.find(';') != std::string::npos) {
        line = line.substr(0, line.find(';'));
      }
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string word;
      while (!iss.eof()) {
        word.clear();
        iss >> word;
        if (word.size() < 2) continue;
        // Trim move numbers from front.
        const auto idx = word.find('.');
        if (idx != std::string::npos) {
          bool all_nums = true;
          for (size_t i = 0; i < idx; i++) {
            if (word[i] < '0' || word[i] > '9') {
              all_nums = false;
              break;
            }
          }
          if (all_nums) {
            word = word.substr(idx + 1);
          }
        }
        // Pure move numbers can be skipped.
        if (word.size() < 2) continue;
        // Ignore score line.
        if (word == "1/2-1/2" || word == "1-0" || word == "0-1" || word == "*")
          continue;
        cur_game_.push_back(SanToMove(word, cur_board_));
        cur_board_.ApplyMove(cur_game_.back());
        // Board ApplyMove wants mirrored for black, but outside code wants
        // normal, so mirror it back again.
        // Check equal to 0 since we've already added the position.
        if ((cur_game_.size() % 2) == 0) {
          cur_game_.back().Mirror();
        }
        cur_board_.Mirror();
      }
    }
    if (started) {
      Flush();
    }
    gzclose(file);
  }
  std::vector<Opening> GetGames() const { return games_; }
  std::vector<Opening>&& ReleaseGames() { return std::move(games_); }
 private:
  void Flush() {
    games_.push_back({cur_startpos_, cur_game_});
    cur_game_.clear();
    cur_board_.SetFromFen(ChessBoard::kStartposFen);
    cur_startpos_ = ChessBoard::kStartposFen;
  }
  Move::Promotion PieceToPromotion(int p) {
    switch (p) {
      case -1:
        return Move::Promotion::None;
      case 2:
        return Move::Promotion::Queen;
      case 3:
        return Move::Promotion::Bishop;
      case 4:
        return Move::Promotion::Knight;
      case 5:
        return Move::Promotion::Rook;
      default:
        // 0 and 1 are pawn and king, which are not legal promotions, other
        // numbers don't correspond to a known piece type.
        CERR << "Unexpected promotion!!";
        throw Exception("Trying to create a move with illegal promotion.");
    }
  }
  Move SanToMove(const std::string& san, const ChessBoard& board) {
    int p = 0;
    size_t idx = 0;
    if (san[0] == 'K') {
      p = 1;
    } else if (san[0] == 'Q') {
      p = 2;
    } else if (san[0] == 'B') {
      p = 3;
    } else if (san[0] == 'N') {
      p = 4;
    } else if (san[0] == 'R') {
      p = 5;
    } else if (san[0] == 'O' && san.size() > 2 && san[1] == '-' &&
               san[2] == 'O') {
      Move m;
      auto king_board = board.kings() & board.ours();
      BoardSquare king_sq(GetLowestBit(king_board.as_int()));
      if (san.size() > 4 && san[3] == '-' && san[4] == 'O') {
        m = Move(BoardSquare(0, king_sq.col()),
                 BoardSquare(0, board.castlings().queenside_rook()));
      } else {
        m = Move(BoardSquare(0, king_sq.col()),
                 BoardSquare(0, board.castlings().kingside_rook()));
      }
      return m;
    }
    if (p != 0) idx++;
    // Formats e4 1e5 de5 d1e5 - with optional x's - followed by =Q for
    // promotions, and even more characters after that also optional.
    int r1 = -1;
    int c1 = -1;
    int r2 = -1;
    int c2 = -1;
    int p2 = -1;
    bool pPending = false;
    for (; idx < san.size(); idx++) {
      if (san[idx] == 'x') continue;
      if (san[idx] == '=') {
        pPending = true;
        continue;
      }
      if (san[idx] >= '1' && san[idx] <= '8') {
        r1 = r2;
        r2 = san[idx] - '1';
        continue;
      }
      if (san[idx] >= 'a' && san[idx] <= 'h') {
        c1 = c2;
        c2 = san[idx] - 'a';
        continue;
      }
      if (pPending) {
        if (san[idx] == 'Q') {
          p2 = 2;
        } else if (san[idx] == 'B') {
          p2 = 3;
        } else if (san[idx] == 'N') {
          p2 = 4;
        } else if (san[idx] == 'R') {
          p2 = 5;
        }
        pPending = false;
        break;
      }
      break;
    }
    if (r1 == -1 || c1 == -1) {
      // Need to find the from cell based on piece.
      int sr1 = r1;
      int sr2 = r2;
      if (board.flipped()) {
        if (sr1 != -1) sr1 = 7 - sr1;
        sr2 = 7 - sr2;
      }
      BitBoard searchBits;
      if (p == 0) {
        searchBits = (board.pawns() & board.ours());
      } else if (p == 1) {
        searchBits = (board.kings() & board.ours());
      } else if (p == 2) {
        searchBits = (board.queens() & board.ours());
      } else if (p == 3) {
        searchBits = (board.bishops() & board.ours());
      } else if (p == 4) {
        searchBits = (board.knights() & board.ours());
      } else if (p == 5) {
        searchBits = (board.rooks() & board.ours());
      }
      auto plm = board.GenerateLegalMoves();
      int pr1 = -1;
      int pc1 = -1;
      for (BoardSquare sq : searchBits) {
        if (sr1 != -1 && sq.row() != sr1) continue;
        if (c1 != -1 && sq.col() != c1) continue;
        if (std::find(plm.begin(), plm.end(),
                      Move(sq, BoardSquare(sr2, c2), PieceToPromotion(p2))) ==
            plm.end()) {
          continue;
        }
        if (pc1 != -1) {
          CERR << "Ambiguous!!";
          throw Exception("Opening book move seems ambiguous.");
        }
        pr1 = sq.row();
        pc1 = sq.col();
      }
      if (pc1 == -1) {
        CERR << "No Match!!";
        throw Exception("Opening book move seems illegal.");
      }
      r1 = pr1;
      c1 = pc1;
      if (board.flipped()) {
        r1 = 7 - r1;
      }
    }
    Move m(BoardSquare(r1, c1), BoardSquare(r2, c2), PieceToPromotion(p2));
    if (board.flipped()) m.Mirror();
    return m;
  }
  ChessBoard cur_board_{ChessBoard::kStartposFen};
  MoveList cur_game_;
  std::string cur_startpos_ = ChessBoard::kStartposFen;
  std::vector<Opening> games_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/pgn.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/stoppers/stoppers.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Combines multiple stoppers into one.
class ChainedSearchStopper : public SearchStopper {
 public:
  ChainedSearchStopper() = default;
  // Calls stoppers one by one until one of them returns true. If one of
  // stoppers modifies hints, next stoppers in the chain see that.
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
  // Can be nullptr, in that canse stopper is not added.
  void AddStopper(std::unique_ptr<SearchStopper> stopper);
  void OnSearchDone(const IterationStats&) override;
 private:
  std::vector<std::unique_ptr<SearchStopper>> stoppers_;
};
// Watches visits (total tree nodes) and predicts remaining visits.
class VisitsStopper : public SearchStopper {
 public:
  VisitsStopper(int64_t limit, bool populate_remaining_playouts)
    : nodes_limit_(limit ? limit : 4000000000ll),
        populate_remaining_playouts_(populate_remaining_playouts) {}
  int64_t GetVisitsLimit() const { return nodes_limit_; }
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
 private:
  const int64_t nodes_limit_;
  const bool populate_remaining_playouts_;
};
// Watches playouts (new tree nodes) and predicts remaining visits.
class PlayoutsStopper : public SearchStopper {
 public:
  PlayoutsStopper(int64_t limit, bool populate_remaining_playouts)
      : nodes_limit_(limit),
        populate_remaining_playouts_(populate_remaining_playouts) {}
  int64_t GetVisitsLimit() const { return nodes_limit_; }
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
 private:
  const int64_t nodes_limit_;
  const bool populate_remaining_playouts_;
};
// Computes tree size which may fit into the memory and limits by that tree
// size.
class MemoryWatchingStopper : public VisitsStopper {
 public:
  // Must be in sync with description at kRamLimitMbId.
  static constexpr size_t kAvgMovesPerPosition = 30;
  MemoryWatchingStopper(int cache_size, int ram_limit_mb,
                        bool populate_remaining_playouts);
};
// Stops after time budget is gone.
class TimeLimitStopper : public SearchStopper {
 public:
  TimeLimitStopper(int64_t time_limit_ms);
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
 protected:
  int64_t GetTimeLimitMs() const;
 private:
  const int64_t time_limit_ms_;
};
// Stops when certain average depth is reached (who needs that?).
class DepthStopper : public SearchStopper {
 public:
  DepthStopper(int depth) : depth_(depth) {}
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
 private:
  const int depth_;
};
// Stops when search doesn't bring required KLD gain.
class KldGainStopper : public SearchStopper {
 public:
  KldGainStopper(float min_gain, int average_interval);
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
 private:
  const double min_gain_;
  const int average_interval_;
  Mutex mutex_;
  std::vector<uint32_t> prev_visits_ GUARDED_BY(mutex_);
  double prev_child_nodes_ GUARDED_BY(mutex_) = 0.0;
};
// Does many things:
// Computes how many nodes are remaining (from remaining time/nodes, scaled by
// smart pruning factor). When this amount of nodes is not enough for second
// best move to potentially become the best one, stop the search.
class SmartPruningStopper : public SearchStopper {
 public:
  SmartPruningStopper(float smart_pruning_factor, int64_t minimum_batches);
  bool ShouldStop(const IterationStats&, StoppersHints*) override;
 private:
  const double smart_pruning_factor_;
  const int64_t minimum_batches_;
  Mutex mutex_;
  std::optional<int64_t> first_eval_time_ GUARDED_BY(mutex_);
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/stoppers.h
// begin /Users/syys/CLionProjects/lc0/src/trainingdata/writer.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
struct V6TrainingData;
class TrainingDataWriter {
 public:
  // Creates a new file to write in data directory. It will has @game_id
  // somewhere in the filename.
  TrainingDataWriter(int game_id);
  ~TrainingDataWriter() {
    if (fout_) Finalize();
  }
  // Writes a chunk.
  void WriteChunk(const V6TrainingData& data);
  // Flushes file and closes it.
  void Finalize();
  // Gets full filename of the file written.
  std::string GetFileName() const { return filename_; }
 private:
  std::string filename_;
  gzFile fout_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/trainingdata/writer.h
// begin /Users/syys/CLionProjects/lc0/src/trainingdata/trainingdata.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
#pragma pack(push, 1)
struct V6TrainingData {
  uint32_t version;
  uint32_t input_format;
  float probabilities[1858];
  uint64_t planes[104];
  uint8_t castling_us_ooo;
  uint8_t castling_us_oo;
  uint8_t castling_them_ooo;
  uint8_t castling_them_oo;
  // For input type 3 contains enpassant column as a mask.
  uint8_t side_to_move_or_enpassant;
  uint8_t rule50_count;
  // Bitfield with the following allocation:
  //  bit 7: side to move (input type 3)
  //  bit 6: position marked for deletion by the rescorer (never set by lc0)
  //  bit 5: game adjudicated (v6)
  //  bit 4: max game length exceeded (v6)
  //  bit 3: best_q is for proven best move (v6)
  //  bit 2: transpose transform (input type 3)
  //  bit 1: mirror transform (input type 3)
  //  bit 0: flip transform (input type 3)
  // In versions prior to v5 this spot contained an unused move count field.
  uint8_t invariance_info;
  // In versions prior to v6 this spot contained thr result as an int8_t.
  uint8_t dummy;
  float root_q;
  float best_q;
  float root_d;
  float best_d;
  float root_m;      // In plies.
  float best_m;      // In plies.
  float plies_left;  // This is the training target for MLH.
  float result_q;
  float result_d;
  float played_q;
  float played_d;
  float played_m;
  // The folowing may be NaN if not found in cache.
  float orig_q;      // For value repair.
  float orig_d;
  float orig_m;
  uint32_t visits;
  // Indices in the probabilities array.
  uint16_t played_idx;
  uint16_t best_idx;
  // Kullback-Leibler divergence between visits and policy (denominator)
  float policy_kld;
  uint32_t reserved;
} PACKED_STRUCT;
static_assert(sizeof(V6TrainingData) == 8356, "Wrong struct size");
#pragma pack(pop)
class V6TrainingDataArray {
 public:
  V6TrainingDataArray(FillEmptyHistory white_fill_empty_history,
                      FillEmptyHistory black_fill_empty_history,
                      pblczero::NetworkFormat::InputFormat input_format)
      : fill_empty_history_{white_fill_empty_history, black_fill_empty_history},
        input_format_(input_format) {}
  // Add a chunk.
  void Add(const Node* node, const PositionHistory& history, Eval best_eval,
           Eval played_eval, bool best_is_proven, Move best_move,
           Move played_move, const NNCacheLock& nneval);
  // Writes training data to a file.
  void Write(TrainingDataWriter* writer, GameResult result,
             bool adjudicated) const;
 private:
  std::vector<V6TrainingData> training_data_;
  FillEmptyHistory fill_empty_history_[2];
  pblczero::NetworkFormat::InputFormat input_format_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/trainingdata/trainingdata.h
// begin /Users/syys/CLionProjects/lc0/src/selfplay/game.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
struct SelfPlayLimits {
  std::int64_t visits = -1;
  std::int64_t playouts = -1;
  std::int64_t movetime = -1;
  std::unique_ptr<ChainedSearchStopper> MakeSearchStopper() const;
};
struct PlayerOptions {
  using OpeningCallback = std::function<void(const Opening&)>;
  // Network to use by the player.
  Network* network;
  // Callback when player moves.
  CallbackUciResponder::BestMoveCallback best_move_callback;
  // Callback when player outputs info.
  CallbackUciResponder::ThinkingCallback info_callback;
  // Callback when player discards a selected move due to low visits.
  OpeningCallback discarded_callback;
  // NNcache to use.
  NNCache* cache;
  // User options dictionary.
  const OptionsDict* uci_options;
  // Limits to use for every move.
  SelfPlayLimits search_limits;
};
// Plays a single game vs itself.
class SelfPlayGame {
 public:
  // Player options may point to the same network/cache/etc.
  // If shared_tree is true, search tree is reused between players.
  // (useful for training games). Otherwise the tree is separate for black
  // and white (useful i.e. when they use different networks).
  SelfPlayGame(PlayerOptions white, PlayerOptions black, bool shared_tree,
               const Opening& opening);
  // Populate command line options that it uses.
  static void PopulateUciParams(OptionsParser* options);
  // Starts the game and blocks until the game is finished.
  void Play(int white_threads, int black_threads, bool training,
            SyzygyTablebase* syzygy_tb, bool enable_resign = true);
  // Aborts the game currently played, doesn't matter if it's synchronous or
  // not.
  void Abort();
  // Writes training data to a file.
  void WriteTrainingData(TrainingDataWriter* writer) const;
  GameResult GetGameResult() const { return game_result_; }
  std::vector<Move> GetMoves() const;
  // Gets the eval which required the biggest swing up to get the final outcome.
  // Eval is the expected outcome in the range 0<->1.
  float GetWorstEvalForWinnerOrDraw() const;
  int move_count_ = 0;
  uint64_t nodes_total_ = 0;
 private:
  // options_[0] is for white player, [1] for black.
  PlayerOptions options_[2];
  // Node tree for player1 and player2. If the tree is shared between players,
  // tree_[0] == tree_[1].
  std::shared_ptr<NodeTree> tree_[2];
  std::string orig_fen_;
  // Search that is currently in progress. Stored in members so that Abort()
  // can stop it.
  std::unique_ptr<Search> search_;
  bool abort_ = false;
  GameResult game_result_ = GameResult::UNDECIDED;
  bool adjudicated_ = false;
  // Track minimum eval for each player so that GetWorstEvalForWinnerOrDraw()
  // can be calculated after end of game.
  float min_eval_[2] = {1.0f, 1.0f};
  // Track the maximum eval for white win, draw, black win for comparison to
  // actual outcome.
  float max_eval_[3] = {0.0f, 0.0f, 0.0f};
  const bool chess960_;
  std::mutex mutex_;
  // Training data to send.
  V6TrainingDataArray training_data_;
  std::unique_ptr<SyzygyTablebase> syzygy_tb_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/selfplay/game.h
// begin /Users/syys/CLionProjects/lc0/src/selfplay/tournament.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Runs many selfplay games, possibly in parallel.
class SelfPlayTournament {
 public:
  SelfPlayTournament(const OptionsDict& options,
                     CallbackUciResponder::BestMoveCallback best_move_info,
                     CallbackUciResponder::ThinkingCallback thinking_info,
                     GameInfo::Callback game_info,
                     TournamentInfo::Callback tournament_info);
  // Populate command line options that it uses.
  static void PopulateOptions(OptionsParser* options);
  // Starts worker threads and exists immediately.
  void StartAsync();
  // Starts tournament and waits until it finishes.
  void RunBlocking();
  // Blocks until all worker threads finish.
  void Wait();
  // Tells worker threads to finish ASAP. Does not block.
  void Abort();
  // Stops any more games from starting, in progress games will complete.
  void Stop();
  // If there are ongoing games, aborts and waits.
  ~SelfPlayTournament();
 private:
  void Worker();
  void PlayOneGame(int game_id);
  Mutex mutex_;
  // Whether first game will be black for player1.
  bool first_game_black_ GUARDED_BY(mutex_) = false;
  std::vector<Opening> discard_pile_ GUARDED_BY(mutex_);
  // Number of games which already started.
  int games_count_ GUARDED_BY(mutex_) = 0;
  bool abort_ GUARDED_BY(mutex_) = false;
  std::vector<Opening> openings_ GUARDED_BY(mutex_);
  // Games in progress. Exposed here to be able to abort them in case if
  // Abort(). Stored as list and not vector so that threads can keep iterators
  // to them and not worry that it becomes invalid.
  std::list<std::unique_ptr<SelfPlayGame>> games_ GUARDED_BY(mutex_);
  // Place to store tournament stats.
  TournamentInfo tournament_info_ GUARDED_BY(mutex_);
  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);
  // Map from the backend configuration to a network.
  std::map<NetworkFactory::BackendConfiguration, std::unique_ptr<Network>>
      networks_;
  std::shared_ptr<NNCache> cache_[2];
  // [player1 or player2][white or black].
  const OptionsDict player_options_[2][2];
  SelfPlayLimits search_limits_[2][2];
  CallbackUciResponder::BestMoveCallback best_move_callback_;
  CallbackUciResponder::ThinkingCallback info_callback_;
  GameInfo::Callback game_callback_;
  TournamentInfo::Callback tournament_callback_;
  const int kTotalGames;
  const bool kShareTree;
  const size_t kParallelism;
  const bool kTraining;
  const float kResignPlaythrough;
  const float kDiscardedStartChance;
  std::unique_ptr<SyzygyTablebase> syzygy_tb_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/selfplay/tournament.h
// begin /Users/syys/CLionProjects/lc0/src/selfplay/loop.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class SelfPlayLoop : public UciLoop {
 public:
  SelfPlayLoop();
  ~SelfPlayLoop();
  void RunLoop() override;
  void CmdStart() override;
  void CmdStop() override;
  void CmdUci() override;
  void CmdSetOption(const std::string& name, const std::string& value,
                    const std::string& context) override;
 private:
  void SendGameInfo(const GameInfo& move);
  void SendTournament(const TournamentInfo& info);
  void EnsureOptionsSent();
  OptionsParser options_;
  std::unique_ptr<SelfPlayTournament> tournament_;
  std::unique_ptr<std::thread> thread_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/selfplay/loop.h
// begin /Users/syys/CLionProjects/lc0/src/utils/commandline.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class CommandLine {
 public:
  CommandLine() = delete;
  // This function must be called before any other.
  static void Init(int argc, const char** argv);
  // Name of the executable filename that was run.
  static const std::string& BinaryName() { return binary_; }
  // Directory where the binary is run. Without trailing slash.
  static std::string BinaryDirectory();
  // If the first command line parameter is @command, remove it and return
  // true. Otherwise return false.
  static bool ConsumeCommand(const std::string& command);
  // Command line arguments.
  static const std::vector<std::string>& Arguments() { return arguments_; }
  static void RegisterMode(const std::string& mode,
                           const std::string& description);
  static const std::vector<std::pair<std::string, std::string>>& GetModes() {
    return modes_;
  }
 private:
  static std::string binary_;
  static std::vector<std::string> arguments_;
  static std::vector<std::pair<std::string, std::string>> modes_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/commandline.h
// begin /Users/syys/CLionProjects/lc0/src/utils/esc_codes.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class EscCodes {
 public:
  EscCodes() = delete;
  // Try to enable ANSI escape sequences for the current terminal.
  static void Init();
  // Supported ANSI escape sequences.
  static const char* Reset() { return enabled_ ? "\033[0m" : ""; }
  static const char* Bold() { return enabled_ ? "\033[1m" : ""; }
  static const char* Underline() { return enabled_ ? "\033[4m" : ""; }
  static const char* Reverse() { return enabled_ ? "\033[7m" : ""; }
  static const char* Normal() { return enabled_ ? "\033[22m" : ""; }
  static const char* NoUnderline() { return enabled_ ? "\033[24m" : ""; }
  static const char* NoReverse() { return enabled_ ? "\033[27m" : ""; }
  static const char* Black() { return enabled_ ? "\033[30m" : ""; }
  static const char* Red() { return enabled_ ? "\033[31m" : ""; }
  static const char* Green() { return enabled_ ? "\033[32m" : ""; }
  static const char* Yellow() { return enabled_ ? "\033[33m" : ""; }
  static const char* Blue() { return enabled_ ? "\033[34m" : ""; }
  static const char* Magenda() { return enabled_ ? "\033[35m" : ""; }
  static const char* Cyan() { return enabled_ ? "\033[36m" : ""; }
  static const char* White() { return enabled_ ? "\033[37m" : ""; }
  static const char* BlackBg() { return enabled_ ? "\033[40m" : ""; }
  static const char* RedBg() { return enabled_ ? "\033[41m" : ""; }
  static const char* GreenBg() { return enabled_ ? "\033[42m" : ""; }
  static const char* YellowBg() { return enabled_ ? "\033[43m" : ""; }
  static const char* BlueBg() { return enabled_ ? "\033[44m" : ""; }
  static const char* MagendaBg() { return enabled_ ? "\033[45m" : ""; }
  static const char* CyanBg() { return enabled_ ? "\033[46m" : ""; }
  static const char* WhiteBg() { return enabled_ ? "\033[47m" : ""; }
 private:
  static bool enabled_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/esc_codes.h
// begin /Users/syys/CLionProjects/lc0/src/version.inc
#define LC0_VERSION_MAJOR 0
#define LC0_VERSION_MINOR 29
#define LC0_VERSION_PATCH 0
#define LC0_VERSION_POSTFIX "dev"

// end of /Users/syys/CLionProjects/lc0/src/version.inc
// begin /Users/syys/CLionProjects/lc0/botzone/meson_conf/build_id.h
#define BUILD_IDENTIFIER "unknown"
// end of /Users/syys/CLionProjects/lc0/botzone/meson_conf/build_id.h
// begin /Users/syys/CLionProjects/lc0/src/version.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
// Versioning is performed according to the standard at <https://semver.org/>
// Creating a new version should be performed using scripts/bumpversion.py.
std::uint32_t GetVersionInt(int major = LC0_VERSION_MAJOR,
                            int minor = LC0_VERSION_MINOR,
                            int patch = LC0_VERSION_PATCH);
std::string GetVersionStr(int major = LC0_VERSION_MAJOR,
                          int minor = LC0_VERSION_MINOR,
                          int patch = LC0_VERSION_PATCH,
                          const std::string& postfix = LC0_VERSION_POSTFIX,
                          const std::string& build_id = BUILD_IDENTIFIER);

// end of /Users/syys/CLionProjects/lc0/src/version.h
// begin /Users/syys/CLionProjects/lc0/src/utils/string.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Joins strings using @delim as delimiter.
std::string StrJoin(const std::vector<std::string>& strings,
                    const std::string& delim = " ");
// Splits strings at whitespace.
std::vector<std::string> StrSplitAtWhitespace(const std::string& str);
// Split string by delimiter.
std::vector<std::string> StrSplit(const std::string& str,
                                  const std::string& delim);
// Parses comma-separated list of integers.
std::vector<int> ParseIntList(const std::string& str);
// Trims a string of whitespace from the start.
std::string LeftTrim(std::string str);
// Trims a string of whitespace from the end.
std::string RightTrim(std::string str);
// Trims a string of whitespace from both ends.
std::string Trim(std::string str);
// Returns whether strings are equal, ignoring case.
bool StringsEqualIgnoreCase(const std::string& a, const std::string& b);
// Flow text into lines of width up to @width.
std::vector<std::string> FlowText(const std::string& src, size_t width);
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/string.h
// begin /Users/syys/CLionProjects/lc0/src/utils/configfile.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class OptionsParser;
class ConfigFile {
 public:
  ConfigFile() = delete;
  // This function must be called after PopulateOptions.
  static bool Init();
  // Returns the command line arguments from the config file.
  static const std::vector<std::string>& Arguments() { return arguments_; }
  // Add the config file parameter to the options dictionary.
  static void PopulateOptions(OptionsParser* options);
 private:
  // Parses the config file into the arguments_ vector.
  static bool ParseFile(std::string& filename);
  // Returns the absolute path to the config file argument given.
  static std::string ProcessConfigFlag(const std::vector<std::string>& args);
  static std::vector<std::string> arguments_;
};
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/configfile.h
// begin /Users/syys/CLionProjects/lc0/src/utils/fastmath.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// These stunts are performed by trained professionals, do not try this at home.
// Fast approximate log2(x). Does no range checking.
// The approximation used here is log2(2^N*(1+f)) ~ N+f*(1+k-k*f) where N is the
// exponent and f the fraction (mantissa), f>=0. The constant k is used to tune
// the approximation accuracy. In the final version some constants were slightly
// modified for better accuracy with 32 bit floating point math.
inline float FastLog2(const float a) {
  uint32_t tmp;
  std::memcpy(&tmp, &a, sizeof(float));
  uint32_t expb = tmp >> 23;
  tmp = (tmp & 0x7fffff) | (0x7f << 23);
  float out;
  std::memcpy(&out, &tmp, sizeof(float));
  out -= 1.0f;
  // Minimize max absolute error.
  return out * (1.3465552f - 0.34655523f * out) - 127 + expb;
}
// Fast approximate 2^x. Does only limited range checking.
// The approximation used here is 2^(N+f) ~ 2^N*(1+f*(1-k+k*f)) where N is the
// integer and f the fractional part, f>=0. The constant k is used to tune the
// approximation accuracy. In the final version some constants were slightly
// modified for better accuracy with 32 bit floating point math.
inline float FastExp2(const float a) {
  int32_t exp;
  if (a < 0) {
    if (a < -126) return 0.0;
    // Not all compilers optimize floor, so we use (a-1) here to round down.
    // This is obviously off-by-one for integer a, but fortunately the error
    // correction term gives the exact value for 1 (by design, for continuity).
    exp = static_cast<int32_t>(a - 1);
  } else {
    exp = static_cast<int32_t>(a);
  }
  float out = a - exp;
  // Minimize max relative error.
  out = 1.0f + out * (0.6602339f + 0.33976606f * out);
  int32_t tmp;
  std::memcpy(&tmp, &out, sizeof(float));
  tmp += static_cast<int32_t>(static_cast<uint32_t>(exp) << 23);
  std::memcpy(&out, &tmp, sizeof(float));
  return out;
}
// Fast approximate ln(x). Does no range checking.
inline float FastLog(const float a) {
  return 0.6931471805599453f * FastLog2(a);
}
// Fast approximate exp(x). Does only limited range checking.
inline float FastExp(const float a) { return FastExp2(1.442695040f * a); }
inline float FastSign(const float a) {
  // Microsoft compiler does not have a builtin for copysign and emits a
  // library call which is too expensive for hot paths.
#if defined(_MSC_VER)
  // This doesn't treat signed 0's the same way that copysign does, but it
  // should be good enough, for our use case.
  return a < 0 ? -1.0f : 1.0f;
#else
  return std::copysign(1.0f, a);
#endif
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/fastmath.h
// begin /Users/syys/CLionProjects/lc0/src/utils/random.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
class Random {
 public:
  static Random& Get();
  double GetDouble(double max_val);
  float GetFloat(float max_val);
  double GetGamma(double alpha, double beta);
  // Both sides are included.
  int GetInt(int min, int max);
  std::string GetString(int length);
  bool GetBool();
  template <class RandomAccessIterator>
  void Shuffle(RandomAccessIterator s, RandomAccessIterator e);
 private:
  Random();
  Mutex mutex_;
  std::mt19937 gen_ GUARDED_BY(mutex_);
};
template <class RandomAccessIterator>
void Random::Shuffle(RandomAccessIterator s, RandomAccessIterator e) {
  Mutex::Lock lock(mutex_);
  std::shuffle(s, e, gen_);
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/random.h
// begin /Users/syys/CLionProjects/lc0/src/utils/filesystem.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Creates directory at a given path. Throws exception if cannot.
// Returns silently if already exists.
void CreateDirectory(const std::string& path);
// Returns list of full paths of regular files in this directory.
// Silently returns empty vector on error.
std::vector<std::string> GetFileList(const std::string& directory);
// Returns size of a file, 0 if file doesn't exist or can't be read.
uint64_t GetFileSize(const std::string& filename);
// Returns modification time of a file, 0 if file doesn't exist or can't be read.
time_t GetFileTime(const std::string& filename);
// Returns the base directory relative to which user specific non-essential data
// files are stored or an empty string if unspecified.
std::string GetUserCacheDirectory();
// Returns the base directory relative to which user specific configuration
// files are stored or an empty string if unspecified.
std::string GetUserConfigDirectory();
// Returns the base directory relative to which user specific data files are
// stored or an empty string if unspecified.
std::string GetUserDataDirectory();
// Returns a vector of base directories to search for configuration files.
std::vector<std::string> GetSystemConfigDirectoryList();
// Returns a vector of base directories to search for data files.
std::vector<std::string> GetSystemDataDirectoryList();
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/filesystem.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/stoppers/factory.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019-2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Populates UCI/command line flags with time management options.
void PopulateTimeManagementOptions(RunType for_what, OptionsParser* options);
// Creates a new time manager for a new search.
std::unique_ptr<TimeManager> MakeTimeManager(const OptionsDict& dict);
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/factory.h
// begin /Users/syys/CLionProjects/lc0/src/neural/onnx/converter.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
// Options to use when converting "old" weights to ONNX weights format.
struct WeightsToOnnxConverterOptions {
  enum class DataType { kFloat32 };
  DataType data_type_ = DataType::kFloat32;
  std::string input_planes_name = "/input/planes";
  std::string output_policy_head = "/output/policy";
  std::string output_wdl = "/output/wdl";
  std::string output_value = "/output/value";
  std::string output_mlh = "/output/mlh";
};
// Converts "classical" weights file to weights file with embedded ONNX model.
pblczero::Net ConvertWeightsToOnnx(const pblczero::Net&,
                                   const WeightsToOnnxConverterOptions&);
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/neural/onnx/converter.h
// begin /Users/syys/CLionProjects/lc0/src/utils/files.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
// Reads (possibly gz-compressed) file to string. Throws on error.
std::string ReadFileToString(const std::string& filename);
// Writes string to file, without compression. Throws on error.
void WriteStringToFile(const std::string& filename,
                         std::string_view  content);
// Writes string to gz-compressed file. Throws on error.
void WriteStringToGzFile(const std::string& filename,
                         std::string_view  content);
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/files.h
// begin /Users/syys/CLionProjects/lc0/src/mcts/stoppers/common.h
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once
namespace lczero {
enum class RunType { kUci, kSelfplay };
void PopulateCommonStopperOptions(RunType for_what, OptionsParser* options);
// Option ID for a cache size. It's used from multiple places and there's no
// really nice place to declare, so let it be here.
extern const OptionId kNNCacheSizeId;
// Populates KLDGain and SmartPruning stoppers.
void PopulateIntrinsicStoppers(ChainedSearchStopper* stopper,
                               const OptionsDict& options);
std::unique_ptr<TimeManager> MakeCommonTimeManager(
    std::unique_ptr<TimeManager> child_manager, const OptionsDict& options,
    int64_t move_overhead);
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/common.h


// ######## end of self header files ######## 


// ######## begin of source files ######## 


// begin of /Users/syys/CLionProjects/lc0/src/main.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
int main(int argc, const char** argv) {
  using namespace lczero;
  EscCodes::Init();
  LOGFILE << "Lc0 started.";
  CERR << EscCodes::Bold() << EscCodes::Red() << "       _";
  CERR << "|   _ | |";
  CERR << "|_ |_ |_|" << EscCodes::Reset() << " v" << GetVersionStr()
       << " built " << __DATE__;
  try {
    Numa::Init();
    Numa::BindThread(0);
    InitializeMagicBitboards();
    CommandLine::Init(argc, argv);
    CommandLine::RegisterMode("uci", "(default) Act as UCI engine");
    CommandLine::RegisterMode("selfplay", "Play games with itself");
    CommandLine::RegisterMode("benchmark", "Quick benchmark");
    CommandLine::RegisterMode("backendbench",
                              "Quick benchmark of backend only");
    CommandLine::RegisterMode("leela2onnx", "Convert Leela network to ONNX.");
    CommandLine::RegisterMode("onnx2leela",
                              "Convert ONNX network to Leela net.");
    CommandLine::RegisterMode("describenet",
                              "Shows details about the Leela network.");
    if (CommandLine::ConsumeCommand("selfplay")) {
      // Selfplay mode.
      SelfPlayLoop loop;
      loop.RunLoop();
    } else if (CommandLine::ConsumeCommand("benchmark")) {
      // Benchmark mode.
      Benchmark benchmark;
      benchmark.Run();
    } else if (CommandLine::ConsumeCommand("backendbench")) {
      // Backend Benchmark mode.
      BackendBenchmark benchmark;
      benchmark.Run();
    } else if (CommandLine::ConsumeCommand("leela2onnx")) {
      lczero::ConvertLeelaToOnnx();
    } else if (CommandLine::ConsumeCommand("onnx2leela")) {
      lczero::ConvertOnnxToLeela();
    } else if (CommandLine::ConsumeCommand("describenet")) {
      lczero::DescribeNetworkCmd();
    } else {
      // Consuming optional "uci" mode.
      CommandLine::ConsumeCommand("uci");
      // Ordinary UCI engine.
      EngineLoop loop;
      loop.RunLoop();
    }
  } catch (std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    abort();
  }
}

// end of /Users/syys/CLionProjects/lc0/src/main.cc

// begin of /Users/syys/CLionProjects/lc0/src/benchmark/backendbench.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const int kDefaultThreads = 1;
const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kBatchesId{"batches", "",
                          "Number of batches to run as a benchmark."};
const OptionId kStartBatchSizeId{"start-batch-size", "",
                                 "Start benchmark from this batch size."};
const OptionId kMaxBatchSizeId{"max-batch-size", "",
                               "Maximum batch size to benchmark."};
const OptionId kBatchStepId{"batch-step", "",
                            "Step of batch size in benchmark."};
const OptionId kFenId{"fen", "", "Benchmark initial position FEN."};
const OptionId kClippyId{"clippy", "", "Enable helpful assistant."};
const OptionId kClippyThresholdId{"clippy-threshold", "",
                                  "Ratio of nps improvement necessary for each "
                                  "doubling of batchsize to be considered "
                                  "best."};
void Clippy(std::string msg) {
  std::cout << "  __" << std::endl;
  std::cout << " /  \\" << std::endl;
  std::cout << " |  |" << std::endl;
  std::cout << " +  +    " << std::string(msg.length() + 2, '_') << std::endl;
  std::cout << "(@)(@) _|" << std::string(msg.length() + 2, ' ') << '|'
            << std::endl;
  std::cout << " |  |  \\  " << msg << " |" << std::endl;
  std::cout << " || |/  |" << std::string(msg.length() + 2, '_') << '|'
            << std::endl;
  std::cout << " || ||" << std::endl;
  std::cout << " |\\_/|" << std::endl;
  std::cout << " \\___/" << std::endl;
}
}  // namespace
void BackendBenchmark::Run() {
  OptionsParser options;
  NetworkFactory::PopulateOptions(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options.Add<IntOption>(kBatchesId, 1, 999999999) = 100;
  options.Add<IntOption>(kStartBatchSizeId, 1, 1024) = 1;
  options.Add<IntOption>(kMaxBatchSizeId, 1, 1024) = 256;
  options.Add<IntOption>(kBatchStepId, 1, 256) = 1;
  options.Add<StringOption>(kFenId) = ChessBoard::kStartposFen;
  options.Add<BoolOption>(kClippyId) = false;
  options.Add<FloatOption>(kClippyThresholdId, 0.0f, 1.0f) = 0.15f;
  if (!options.ProcessAllFlags()) return;
  try {
    auto option_dict = options.GetOptionsDict();
    auto network = NetworkFactory::LoadNetwork(option_dict);
    NodeTree tree;
    tree.ResetToPosition(option_dict.Get<std::string>(kFenId), {});
    // Do any backend initialization outside the loop.
    auto warmup = network->NewComputation();
    warmup->AddInput(EncodePositionForNN(
        network->GetCapabilities().input_format, tree.GetPositionHistory(), 8,
        FillEmptyHistory::ALWAYS, nullptr));
    warmup->ComputeBlocking();
    const int batches = option_dict.Get<int>(kBatchesId);
    int best = 1;
    float best_nps = 0.0f;
    std::optional<std::chrono::time_point<std::chrono::steady_clock>> pending;
    for (int i = option_dict.Get<int>(kStartBatchSizeId);
         i <= option_dict.Get<int>(kMaxBatchSizeId);
         i += option_dict.Get<int>(kBatchStepId)) {
      const auto start = std::chrono::steady_clock::now();
      // TODO: support threads not equal to 1 to be able to more sensibly test
      // multiplexing backend.
      for (int j = 0; j < batches; j++) {
        // Put i copies of tree root node into computation and compute.
        auto computation = network->NewComputation();
        for (int k = 0; k < i; k++) {
          computation->AddInput(EncodePositionForNN(
              network->GetCapabilities().input_format,
              tree.GetPositionHistory(), 8, FillEmptyHistory::ALWAYS, nullptr));
        }
        computation->ComputeBlocking();
      }
      const auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> time = end - start;
      const auto nps = i * batches / time.count();
      std::cout << "Benchmark batch size " << i
                << " with inference average time "
                << time.count() / batches * 1000 << "ms - throughput " << nps
                << " nps." << std::endl;
      if (option_dict.Get<bool>(kClippyId)) {
        const float threshold = option_dict.Get<float>(kClippyThresholdId);
        if (nps > best_nps &&
            threshold * (i - best) * best_nps < (nps - best_nps) * best) {
          best_nps = nps;
          best = i;
          if (!pending) {
            pending = std::chrono::steady_clock::now();
          }
        }
        if (pending) {
          time = std::chrono::steady_clock::now() - *pending;
          if (time.count() > 10) {
            Clippy(
                std::to_string(best) +
                " looks like the best minibatch-size for this net (so far).");
            pending.reset();
          }
        }
      }
    }
    if (option_dict.Get<bool>(kClippyId)) {
      Clippy(std::to_string(best) +
             " looks like the best minibatch-size for this net.");
    }
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/benchmark/backendbench.cc

// begin of /Users/syys/CLionProjects/lc0/src/chess/bitboard.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const Move kIdxToMove[] = {
    "a1b1",  "a1c1",  "a1d1",  "a1e1",  "a1f1",  "a1g1",  "a1h1",  "a1a2",
    "a1b2",  "a1c2",  "a1a3",  "a1b3",  "a1c3",  "a1a4",  "a1d4",  "a1a5",
    "a1e5",  "a1a6",  "a1f6",  "a1a7",  "a1g7",  "a1a8",  "a1h8",  "b1a1",
    "b1c1",  "b1d1",  "b1e1",  "b1f1",  "b1g1",  "b1h1",  "b1a2",  "b1b2",
    "b1c2",  "b1d2",  "b1a3",  "b1b3",  "b1c3",  "b1d3",  "b1b4",  "b1e4",
    "b1b5",  "b1f5",  "b1b6",  "b1g6",  "b1b7",  "b1h7",  "b1b8",  "c1a1",
    "c1b1",  "c1d1",  "c1e1",  "c1f1",  "c1g1",  "c1h1",  "c1a2",  "c1b2",
    "c1c2",  "c1d2",  "c1e2",  "c1a3",  "c1b3",  "c1c3",  "c1d3",  "c1e3",
    "c1c4",  "c1f4",  "c1c5",  "c1g5",  "c1c6",  "c1h6",  "c1c7",  "c1c8",
    "d1a1",  "d1b1",  "d1c1",  "d1e1",  "d1f1",  "d1g1",  "d1h1",  "d1b2",
    "d1c2",  "d1d2",  "d1e2",  "d1f2",  "d1b3",  "d1c3",  "d1d3",  "d1e3",
    "d1f3",  "d1a4",  "d1d4",  "d1g4",  "d1d5",  "d1h5",  "d1d6",  "d1d7",
    "d1d8",  "e1a1",  "e1b1",  "e1c1",  "e1d1",  "e1f1",  "e1g1",  "e1h1",
    "e1c2",  "e1d2",  "e1e2",  "e1f2",  "e1g2",  "e1c3",  "e1d3",  "e1e3",
    "e1f3",  "e1g3",  "e1b4",  "e1e4",  "e1h4",  "e1a5",  "e1e5",  "e1e6",
    "e1e7",  "e1e8",  "f1a1",  "f1b1",  "f1c1",  "f1d1",  "f1e1",  "f1g1",
    "f1h1",  "f1d2",  "f1e2",  "f1f2",  "f1g2",  "f1h2",  "f1d3",  "f1e3",
    "f1f3",  "f1g3",  "f1h3",  "f1c4",  "f1f4",  "f1b5",  "f1f5",  "f1a6",
    "f1f6",  "f1f7",  "f1f8",  "g1a1",  "g1b1",  "g1c1",  "g1d1",  "g1e1",
    "g1f1",  "g1h1",  "g1e2",  "g1f2",  "g1g2",  "g1h2",  "g1e3",  "g1f3",
    "g1g3",  "g1h3",  "g1d4",  "g1g4",  "g1c5",  "g1g5",  "g1b6",  "g1g6",
    "g1a7",  "g1g7",  "g1g8",  "h1a1",  "h1b1",  "h1c1",  "h1d1",  "h1e1",
    "h1f1",  "h1g1",  "h1f2",  "h1g2",  "h1h2",  "h1f3",  "h1g3",  "h1h3",
    "h1e4",  "h1h4",  "h1d5",  "h1h5",  "h1c6",  "h1h6",  "h1b7",  "h1h7",
    "h1a8",  "h1h8",  "a2a1",  "a2b1",  "a2c1",  "a2b2",  "a2c2",  "a2d2",
    "a2e2",  "a2f2",  "a2g2",  "a2h2",  "a2a3",  "a2b3",  "a2c3",  "a2a4",
    "a2b4",  "a2c4",  "a2a5",  "a2d5",  "a2a6",  "a2e6",  "a2a7",  "a2f7",
    "a2a8",  "a2g8",  "b2a1",  "b2b1",  "b2c1",  "b2d1",  "b2a2",  "b2c2",
    "b2d2",  "b2e2",  "b2f2",  "b2g2",  "b2h2",  "b2a3",  "b2b3",  "b2c3",
    "b2d3",  "b2a4",  "b2b4",  "b2c4",  "b2d4",  "b2b5",  "b2e5",  "b2b6",
    "b2f6",  "b2b7",  "b2g7",  "b2b8",  "b2h8",  "c2a1",  "c2b1",  "c2c1",
    "c2d1",  "c2e1",  "c2a2",  "c2b2",  "c2d2",  "c2e2",  "c2f2",  "c2g2",
    "c2h2",  "c2a3",  "c2b3",  "c2c3",  "c2d3",  "c2e3",  "c2a4",  "c2b4",
    "c2c4",  "c2d4",  "c2e4",  "c2c5",  "c2f5",  "c2c6",  "c2g6",  "c2c7",
    "c2h7",  "c2c8",  "d2b1",  "d2c1",  "d2d1",  "d2e1",  "d2f1",  "d2a2",
    "d2b2",  "d2c2",  "d2e2",  "d2f2",  "d2g2",  "d2h2",  "d2b3",  "d2c3",
    "d2d3",  "d2e3",  "d2f3",  "d2b4",  "d2c4",  "d2d4",  "d2e4",  "d2f4",
    "d2a5",  "d2d5",  "d2g5",  "d2d6",  "d2h6",  "d2d7",  "d2d8",  "e2c1",
    "e2d1",  "e2e1",  "e2f1",  "e2g1",  "e2a2",  "e2b2",  "e2c2",  "e2d2",
    "e2f2",  "e2g2",  "e2h2",  "e2c3",  "e2d3",  "e2e3",  "e2f3",  "e2g3",
    "e2c4",  "e2d4",  "e2e4",  "e2f4",  "e2g4",  "e2b5",  "e2e5",  "e2h5",
    "e2a6",  "e2e6",  "e2e7",  "e2e8",  "f2d1",  "f2e1",  "f2f1",  "f2g1",
    "f2h1",  "f2a2",  "f2b2",  "f2c2",  "f2d2",  "f2e2",  "f2g2",  "f2h2",
    "f2d3",  "f2e3",  "f2f3",  "f2g3",  "f2h3",  "f2d4",  "f2e4",  "f2f4",
    "f2g4",  "f2h4",  "f2c5",  "f2f5",  "f2b6",  "f2f6",  "f2a7",  "f2f7",
    "f2f8",  "g2e1",  "g2f1",  "g2g1",  "g2h1",  "g2a2",  "g2b2",  "g2c2",
    "g2d2",  "g2e2",  "g2f2",  "g2h2",  "g2e3",  "g2f3",  "g2g3",  "g2h3",
    "g2e4",  "g2f4",  "g2g4",  "g2h4",  "g2d5",  "g2g5",  "g2c6",  "g2g6",
    "g2b7",  "g2g7",  "g2a8",  "g2g8",  "h2f1",  "h2g1",  "h2h1",  "h2a2",
    "h2b2",  "h2c2",  "h2d2",  "h2e2",  "h2f2",  "h2g2",  "h2f3",  "h2g3",
    "h2h3",  "h2f4",  "h2g4",  "h2h4",  "h2e5",  "h2h5",  "h2d6",  "h2h6",
    "h2c7",  "h2h7",  "h2b8",  "h2h8",  "a3a1",  "a3b1",  "a3c1",  "a3a2",
    "a3b2",  "a3c2",  "a3b3",  "a3c3",  "a3d3",  "a3e3",  "a3f3",  "a3g3",
    "a3h3",  "a3a4",  "a3b4",  "a3c4",  "a3a5",  "a3b5",  "a3c5",  "a3a6",
    "a3d6",  "a3a7",  "a3e7",  "a3a8",  "a3f8",  "b3a1",  "b3b1",  "b3c1",
    "b3d1",  "b3a2",  "b3b2",  "b3c2",  "b3d2",  "b3a3",  "b3c3",  "b3d3",
    "b3e3",  "b3f3",  "b3g3",  "b3h3",  "b3a4",  "b3b4",  "b3c4",  "b3d4",
    "b3a5",  "b3b5",  "b3c5",  "b3d5",  "b3b6",  "b3e6",  "b3b7",  "b3f7",
    "b3b8",  "b3g8",  "c3a1",  "c3b1",  "c3c1",  "c3d1",  "c3e1",  "c3a2",
    "c3b2",  "c3c2",  "c3d2",  "c3e2",  "c3a3",  "c3b3",  "c3d3",  "c3e3",
    "c3f3",  "c3g3",  "c3h3",  "c3a4",  "c3b4",  "c3c4",  "c3d4",  "c3e4",
    "c3a5",  "c3b5",  "c3c5",  "c3d5",  "c3e5",  "c3c6",  "c3f6",  "c3c7",
    "c3g7",  "c3c8",  "c3h8",  "d3b1",  "d3c1",  "d3d1",  "d3e1",  "d3f1",
    "d3b2",  "d3c2",  "d3d2",  "d3e2",  "d3f2",  "d3a3",  "d3b3",  "d3c3",
    "d3e3",  "d3f3",  "d3g3",  "d3h3",  "d3b4",  "d3c4",  "d3d4",  "d3e4",
    "d3f4",  "d3b5",  "d3c5",  "d3d5",  "d3e5",  "d3f5",  "d3a6",  "d3d6",
    "d3g6",  "d3d7",  "d3h7",  "d3d8",  "e3c1",  "e3d1",  "e3e1",  "e3f1",
    "e3g1",  "e3c2",  "e3d2",  "e3e2",  "e3f2",  "e3g2",  "e3a3",  "e3b3",
    "e3c3",  "e3d3",  "e3f3",  "e3g3",  "e3h3",  "e3c4",  "e3d4",  "e3e4",
    "e3f4",  "e3g4",  "e3c5",  "e3d5",  "e3e5",  "e3f5",  "e3g5",  "e3b6",
    "e3e6",  "e3h6",  "e3a7",  "e3e7",  "e3e8",  "f3d1",  "f3e1",  "f3f1",
    "f3g1",  "f3h1",  "f3d2",  "f3e2",  "f3f2",  "f3g2",  "f3h2",  "f3a3",
    "f3b3",  "f3c3",  "f3d3",  "f3e3",  "f3g3",  "f3h3",  "f3d4",  "f3e4",
    "f3f4",  "f3g4",  "f3h4",  "f3d5",  "f3e5",  "f3f5",  "f3g5",  "f3h5",
    "f3c6",  "f3f6",  "f3b7",  "f3f7",  "f3a8",  "f3f8",  "g3e1",  "g3f1",
    "g3g1",  "g3h1",  "g3e2",  "g3f2",  "g3g2",  "g3h2",  "g3a3",  "g3b3",
    "g3c3",  "g3d3",  "g3e3",  "g3f3",  "g3h3",  "g3e4",  "g3f4",  "g3g4",
    "g3h4",  "g3e5",  "g3f5",  "g3g5",  "g3h5",  "g3d6",  "g3g6",  "g3c7",
    "g3g7",  "g3b8",  "g3g8",  "h3f1",  "h3g1",  "h3h1",  "h3f2",  "h3g2",
    "h3h2",  "h3a3",  "h3b3",  "h3c3",  "h3d3",  "h3e3",  "h3f3",  "h3g3",
    "h3f4",  "h3g4",  "h3h4",  "h3f5",  "h3g5",  "h3h5",  "h3e6",  "h3h6",
    "h3d7",  "h3h7",  "h3c8",  "h3h8",  "a4a1",  "a4d1",  "a4a2",  "a4b2",
    "a4c2",  "a4a3",  "a4b3",  "a4c3",  "a4b4",  "a4c4",  "a4d4",  "a4e4",
    "a4f4",  "a4g4",  "a4h4",  "a4a5",  "a4b5",  "a4c5",  "a4a6",  "a4b6",
    "a4c6",  "a4a7",  "a4d7",  "a4a8",  "a4e8",  "b4b1",  "b4e1",  "b4a2",
    "b4b2",  "b4c2",  "b4d2",  "b4a3",  "b4b3",  "b4c3",  "b4d3",  "b4a4",
    "b4c4",  "b4d4",  "b4e4",  "b4f4",  "b4g4",  "b4h4",  "b4a5",  "b4b5",
    "b4c5",  "b4d5",  "b4a6",  "b4b6",  "b4c6",  "b4d6",  "b4b7",  "b4e7",
    "b4b8",  "b4f8",  "c4c1",  "c4f1",  "c4a2",  "c4b2",  "c4c2",  "c4d2",
    "c4e2",  "c4a3",  "c4b3",  "c4c3",  "c4d3",  "c4e3",  "c4a4",  "c4b4",
    "c4d4",  "c4e4",  "c4f4",  "c4g4",  "c4h4",  "c4a5",  "c4b5",  "c4c5",
    "c4d5",  "c4e5",  "c4a6",  "c4b6",  "c4c6",  "c4d6",  "c4e6",  "c4c7",
    "c4f7",  "c4c8",  "c4g8",  "d4a1",  "d4d1",  "d4g1",  "d4b2",  "d4c2",
    "d4d2",  "d4e2",  "d4f2",  "d4b3",  "d4c3",  "d4d3",  "d4e3",  "d4f3",
    "d4a4",  "d4b4",  "d4c4",  "d4e4",  "d4f4",  "d4g4",  "d4h4",  "d4b5",
    "d4c5",  "d4d5",  "d4e5",  "d4f5",  "d4b6",  "d4c6",  "d4d6",  "d4e6",
    "d4f6",  "d4a7",  "d4d7",  "d4g7",  "d4d8",  "d4h8",  "e4b1",  "e4e1",
    "e4h1",  "e4c2",  "e4d2",  "e4e2",  "e4f2",  "e4g2",  "e4c3",  "e4d3",
    "e4e3",  "e4f3",  "e4g3",  "e4a4",  "e4b4",  "e4c4",  "e4d4",  "e4f4",
    "e4g4",  "e4h4",  "e4c5",  "e4d5",  "e4e5",  "e4f5",  "e4g5",  "e4c6",
    "e4d6",  "e4e6",  "e4f6",  "e4g6",  "e4b7",  "e4e7",  "e4h7",  "e4a8",
    "e4e8",  "f4c1",  "f4f1",  "f4d2",  "f4e2",  "f4f2",  "f4g2",  "f4h2",
    "f4d3",  "f4e3",  "f4f3",  "f4g3",  "f4h3",  "f4a4",  "f4b4",  "f4c4",
    "f4d4",  "f4e4",  "f4g4",  "f4h4",  "f4d5",  "f4e5",  "f4f5",  "f4g5",
    "f4h5",  "f4d6",  "f4e6",  "f4f6",  "f4g6",  "f4h6",  "f4c7",  "f4f7",
    "f4b8",  "f4f8",  "g4d1",  "g4g1",  "g4e2",  "g4f2",  "g4g2",  "g4h2",
    "g4e3",  "g4f3",  "g4g3",  "g4h3",  "g4a4",  "g4b4",  "g4c4",  "g4d4",
    "g4e4",  "g4f4",  "g4h4",  "g4e5",  "g4f5",  "g4g5",  "g4h5",  "g4e6",
    "g4f6",  "g4g6",  "g4h6",  "g4d7",  "g4g7",  "g4c8",  "g4g8",  "h4e1",
    "h4h1",  "h4f2",  "h4g2",  "h4h2",  "h4f3",  "h4g3",  "h4h3",  "h4a4",
    "h4b4",  "h4c4",  "h4d4",  "h4e4",  "h4f4",  "h4g4",  "h4f5",  "h4g5",
    "h4h5",  "h4f6",  "h4g6",  "h4h6",  "h4e7",  "h4h7",  "h4d8",  "h4h8",
    "a5a1",  "a5e1",  "a5a2",  "a5d2",  "a5a3",  "a5b3",  "a5c3",  "a5a4",
    "a5b4",  "a5c4",  "a5b5",  "a5c5",  "a5d5",  "a5e5",  "a5f5",  "a5g5",
    "a5h5",  "a5a6",  "a5b6",  "a5c6",  "a5a7",  "a5b7",  "a5c7",  "a5a8",
    "a5d8",  "b5b1",  "b5f1",  "b5b2",  "b5e2",  "b5a3",  "b5b3",  "b5c3",
    "b5d3",  "b5a4",  "b5b4",  "b5c4",  "b5d4",  "b5a5",  "b5c5",  "b5d5",
    "b5e5",  "b5f5",  "b5g5",  "b5h5",  "b5a6",  "b5b6",  "b5c6",  "b5d6",
    "b5a7",  "b5b7",  "b5c7",  "b5d7",  "b5b8",  "b5e8",  "c5c1",  "c5g1",
    "c5c2",  "c5f2",  "c5a3",  "c5b3",  "c5c3",  "c5d3",  "c5e3",  "c5a4",
    "c5b4",  "c5c4",  "c5d4",  "c5e4",  "c5a5",  "c5b5",  "c5d5",  "c5e5",
    "c5f5",  "c5g5",  "c5h5",  "c5a6",  "c5b6",  "c5c6",  "c5d6",  "c5e6",
    "c5a7",  "c5b7",  "c5c7",  "c5d7",  "c5e7",  "c5c8",  "c5f8",  "d5d1",
    "d5h1",  "d5a2",  "d5d2",  "d5g2",  "d5b3",  "d5c3",  "d5d3",  "d5e3",
    "d5f3",  "d5b4",  "d5c4",  "d5d4",  "d5e4",  "d5f4",  "d5a5",  "d5b5",
    "d5c5",  "d5e5",  "d5f5",  "d5g5",  "d5h5",  "d5b6",  "d5c6",  "d5d6",
    "d5e6",  "d5f6",  "d5b7",  "d5c7",  "d5d7",  "d5e7",  "d5f7",  "d5a8",
    "d5d8",  "d5g8",  "e5a1",  "e5e1",  "e5b2",  "e5e2",  "e5h2",  "e5c3",
    "e5d3",  "e5e3",  "e5f3",  "e5g3",  "e5c4",  "e5d4",  "e5e4",  "e5f4",
    "e5g4",  "e5a5",  "e5b5",  "e5c5",  "e5d5",  "e5f5",  "e5g5",  "e5h5",
    "e5c6",  "e5d6",  "e5e6",  "e5f6",  "e5g6",  "e5c7",  "e5d7",  "e5e7",
    "e5f7",  "e5g7",  "e5b8",  "e5e8",  "e5h8",  "f5b1",  "f5f1",  "f5c2",
    "f5f2",  "f5d3",  "f5e3",  "f5f3",  "f5g3",  "f5h3",  "f5d4",  "f5e4",
    "f5f4",  "f5g4",  "f5h4",  "f5a5",  "f5b5",  "f5c5",  "f5d5",  "f5e5",
    "f5g5",  "f5h5",  "f5d6",  "f5e6",  "f5f6",  "f5g6",  "f5h6",  "f5d7",
    "f5e7",  "f5f7",  "f5g7",  "f5h7",  "f5c8",  "f5f8",  "g5c1",  "g5g1",
    "g5d2",  "g5g2",  "g5e3",  "g5f3",  "g5g3",  "g5h3",  "g5e4",  "g5f4",
    "g5g4",  "g5h4",  "g5a5",  "g5b5",  "g5c5",  "g5d5",  "g5e5",  "g5f5",
    "g5h5",  "g5e6",  "g5f6",  "g5g6",  "g5h6",  "g5e7",  "g5f7",  "g5g7",
    "g5h7",  "g5d8",  "g5g8",  "h5d1",  "h5h1",  "h5e2",  "h5h2",  "h5f3",
    "h5g3",  "h5h3",  "h5f4",  "h5g4",  "h5h4",  "h5a5",  "h5b5",  "h5c5",
    "h5d5",  "h5e5",  "h5f5",  "h5g5",  "h5f6",  "h5g6",  "h5h6",  "h5f7",
    "h5g7",  "h5h7",  "h5e8",  "h5h8",  "a6a1",  "a6f1",  "a6a2",  "a6e2",
    "a6a3",  "a6d3",  "a6a4",  "a6b4",  "a6c4",  "a6a5",  "a6b5",  "a6c5",
    "a6b6",  "a6c6",  "a6d6",  "a6e6",  "a6f6",  "a6g6",  "a6h6",  "a6a7",
    "a6b7",  "a6c7",  "a6a8",  "a6b8",  "a6c8",  "b6b1",  "b6g1",  "b6b2",
    "b6f2",  "b6b3",  "b6e3",  "b6a4",  "b6b4",  "b6c4",  "b6d4",  "b6a5",
    "b6b5",  "b6c5",  "b6d5",  "b6a6",  "b6c6",  "b6d6",  "b6e6",  "b6f6",
    "b6g6",  "b6h6",  "b6a7",  "b6b7",  "b6c7",  "b6d7",  "b6a8",  "b6b8",
    "b6c8",  "b6d8",  "c6c1",  "c6h1",  "c6c2",  "c6g2",  "c6c3",  "c6f3",
    "c6a4",  "c6b4",  "c6c4",  "c6d4",  "c6e4",  "c6a5",  "c6b5",  "c6c5",
    "c6d5",  "c6e5",  "c6a6",  "c6b6",  "c6d6",  "c6e6",  "c6f6",  "c6g6",
    "c6h6",  "c6a7",  "c6b7",  "c6c7",  "c6d7",  "c6e7",  "c6a8",  "c6b8",
    "c6c8",  "c6d8",  "c6e8",  "d6d1",  "d6d2",  "d6h2",  "d6a3",  "d6d3",
    "d6g3",  "d6b4",  "d6c4",  "d6d4",  "d6e4",  "d6f4",  "d6b5",  "d6c5",
    "d6d5",  "d6e5",  "d6f5",  "d6a6",  "d6b6",  "d6c6",  "d6e6",  "d6f6",
    "d6g6",  "d6h6",  "d6b7",  "d6c7",  "d6d7",  "d6e7",  "d6f7",  "d6b8",
    "d6c8",  "d6d8",  "d6e8",  "d6f8",  "e6e1",  "e6a2",  "e6e2",  "e6b3",
    "e6e3",  "e6h3",  "e6c4",  "e6d4",  "e6e4",  "e6f4",  "e6g4",  "e6c5",
    "e6d5",  "e6e5",  "e6f5",  "e6g5",  "e6a6",  "e6b6",  "e6c6",  "e6d6",
    "e6f6",  "e6g6",  "e6h6",  "e6c7",  "e6d7",  "e6e7",  "e6f7",  "e6g7",
    "e6c8",  "e6d8",  "e6e8",  "e6f8",  "e6g8",  "f6a1",  "f6f1",  "f6b2",
    "f6f2",  "f6c3",  "f6f3",  "f6d4",  "f6e4",  "f6f4",  "f6g4",  "f6h4",
    "f6d5",  "f6e5",  "f6f5",  "f6g5",  "f6h5",  "f6a6",  "f6b6",  "f6c6",
    "f6d6",  "f6e6",  "f6g6",  "f6h6",  "f6d7",  "f6e7",  "f6f7",  "f6g7",
    "f6h7",  "f6d8",  "f6e8",  "f6f8",  "f6g8",  "f6h8",  "g6b1",  "g6g1",
    "g6c2",  "g6g2",  "g6d3",  "g6g3",  "g6e4",  "g6f4",  "g6g4",  "g6h4",
    "g6e5",  "g6f5",  "g6g5",  "g6h5",  "g6a6",  "g6b6",  "g6c6",  "g6d6",
    "g6e6",  "g6f6",  "g6h6",  "g6e7",  "g6f7",  "g6g7",  "g6h7",  "g6e8",
    "g6f8",  "g6g8",  "g6h8",  "h6c1",  "h6h1",  "h6d2",  "h6h2",  "h6e3",
    "h6h3",  "h6f4",  "h6g4",  "h6h4",  "h6f5",  "h6g5",  "h6h5",  "h6a6",
    "h6b6",  "h6c6",  "h6d6",  "h6e6",  "h6f6",  "h6g6",  "h6f7",  "h6g7",
    "h6h7",  "h6f8",  "h6g8",  "h6h8",  "a7a1",  "a7g1",  "a7a2",  "a7f2",
    "a7a3",  "a7e3",  "a7a4",  "a7d4",  "a7a5",  "a7b5",  "a7c5",  "a7a6",
    "a7b6",  "a7c6",  "a7b7",  "a7c7",  "a7d7",  "a7e7",  "a7f7",  "a7g7",
    "a7h7",  "a7a8",  "a7b8",  "a7c8",  "b7b1",  "b7h1",  "b7b2",  "b7g2",
    "b7b3",  "b7f3",  "b7b4",  "b7e4",  "b7a5",  "b7b5",  "b7c5",  "b7d5",
    "b7a6",  "b7b6",  "b7c6",  "b7d6",  "b7a7",  "b7c7",  "b7d7",  "b7e7",
    "b7f7",  "b7g7",  "b7h7",  "b7a8",  "b7b8",  "b7c8",  "b7d8",  "c7c1",
    "c7c2",  "c7h2",  "c7c3",  "c7g3",  "c7c4",  "c7f4",  "c7a5",  "c7b5",
    "c7c5",  "c7d5",  "c7e5",  "c7a6",  "c7b6",  "c7c6",  "c7d6",  "c7e6",
    "c7a7",  "c7b7",  "c7d7",  "c7e7",  "c7f7",  "c7g7",  "c7h7",  "c7a8",
    "c7b8",  "c7c8",  "c7d8",  "c7e8",  "d7d1",  "d7d2",  "d7d3",  "d7h3",
    "d7a4",  "d7d4",  "d7g4",  "d7b5",  "d7c5",  "d7d5",  "d7e5",  "d7f5",
    "d7b6",  "d7c6",  "d7d6",  "d7e6",  "d7f6",  "d7a7",  "d7b7",  "d7c7",
    "d7e7",  "d7f7",  "d7g7",  "d7h7",  "d7b8",  "d7c8",  "d7d8",  "d7e8",
    "d7f8",  "e7e1",  "e7e2",  "e7a3",  "e7e3",  "e7b4",  "e7e4",  "e7h4",
    "e7c5",  "e7d5",  "e7e5",  "e7f5",  "e7g5",  "e7c6",  "e7d6",  "e7e6",
    "e7f6",  "e7g6",  "e7a7",  "e7b7",  "e7c7",  "e7d7",  "e7f7",  "e7g7",
    "e7h7",  "e7c8",  "e7d8",  "e7e8",  "e7f8",  "e7g8",  "f7f1",  "f7a2",
    "f7f2",  "f7b3",  "f7f3",  "f7c4",  "f7f4",  "f7d5",  "f7e5",  "f7f5",
    "f7g5",  "f7h5",  "f7d6",  "f7e6",  "f7f6",  "f7g6",  "f7h6",  "f7a7",
    "f7b7",  "f7c7",  "f7d7",  "f7e7",  "f7g7",  "f7h7",  "f7d8",  "f7e8",
    "f7f8",  "f7g8",  "f7h8",  "g7a1",  "g7g1",  "g7b2",  "g7g2",  "g7c3",
    "g7g3",  "g7d4",  "g7g4",  "g7e5",  "g7f5",  "g7g5",  "g7h5",  "g7e6",
    "g7f6",  "g7g6",  "g7h6",  "g7a7",  "g7b7",  "g7c7",  "g7d7",  "g7e7",
    "g7f7",  "g7h7",  "g7e8",  "g7f8",  "g7g8",  "g7h8",  "h7b1",  "h7h1",
    "h7c2",  "h7h2",  "h7d3",  "h7h3",  "h7e4",  "h7h4",  "h7f5",  "h7g5",
    "h7h5",  "h7f6",  "h7g6",  "h7h6",  "h7a7",  "h7b7",  "h7c7",  "h7d7",
    "h7e7",  "h7f7",  "h7g7",  "h7f8",  "h7g8",  "h7h8",  "a8a1",  "a8h1",
    "a8a2",  "a8g2",  "a8a3",  "a8f3",  "a8a4",  "a8e4",  "a8a5",  "a8d5",
    "a8a6",  "a8b6",  "a8c6",  "a8a7",  "a8b7",  "a8c7",  "a8b8",  "a8c8",
    "a8d8",  "a8e8",  "a8f8",  "a8g8",  "a8h8",  "b8b1",  "b8b2",  "b8h2",
    "b8b3",  "b8g3",  "b8b4",  "b8f4",  "b8b5",  "b8e5",  "b8a6",  "b8b6",
    "b8c6",  "b8d6",  "b8a7",  "b8b7",  "b8c7",  "b8d7",  "b8a8",  "b8c8",
    "b8d8",  "b8e8",  "b8f8",  "b8g8",  "b8h8",  "c8c1",  "c8c2",  "c8c3",
    "c8h3",  "c8c4",  "c8g4",  "c8c5",  "c8f5",  "c8a6",  "c8b6",  "c8c6",
    "c8d6",  "c8e6",  "c8a7",  "c8b7",  "c8c7",  "c8d7",  "c8e7",  "c8a8",
    "c8b8",  "c8d8",  "c8e8",  "c8f8",  "c8g8",  "c8h8",  "d8d1",  "d8d2",
    "d8d3",  "d8d4",  "d8h4",  "d8a5",  "d8d5",  "d8g5",  "d8b6",  "d8c6",
    "d8d6",  "d8e6",  "d8f6",  "d8b7",  "d8c7",  "d8d7",  "d8e7",  "d8f7",
    "d8a8",  "d8b8",  "d8c8",  "d8e8",  "d8f8",  "d8g8",  "d8h8",  "e8e1",
    "e8e2",  "e8e3",  "e8a4",  "e8e4",  "e8b5",  "e8e5",  "e8h5",  "e8c6",
    "e8d6",  "e8e6",  "e8f6",  "e8g6",  "e8c7",  "e8d7",  "e8e7",  "e8f7",
    "e8g7",  "e8a8",  "e8b8",  "e8c8",  "e8d8",  "e8f8",  "e8g8",  "e8h8",
    "f8f1",  "f8f2",  "f8a3",  "f8f3",  "f8b4",  "f8f4",  "f8c5",  "f8f5",
    "f8d6",  "f8e6",  "f8f6",  "f8g6",  "f8h6",  "f8d7",  "f8e7",  "f8f7",
    "f8g7",  "f8h7",  "f8a8",  "f8b8",  "f8c8",  "f8d8",  "f8e8",  "f8g8",
    "f8h8",  "g8g1",  "g8a2",  "g8g2",  "g8b3",  "g8g3",  "g8c4",  "g8g4",
    "g8d5",  "g8g5",  "g8e6",  "g8f6",  "g8g6",  "g8h6",  "g8e7",  "g8f7",
    "g8g7",  "g8h7",  "g8a8",  "g8b8",  "g8c8",  "g8d8",  "g8e8",  "g8f8",
    "g8h8",  "h8a1",  "h8h1",  "h8b2",  "h8h2",  "h8c3",  "h8h3",  "h8d4",
    "h8h4",  "h8e5",  "h8h5",  "h8f6",  "h8g6",  "h8h6",  "h8f7",  "h8g7",
    "h8h7",  "h8a8",  "h8b8",  "h8c8",  "h8d8",  "h8e8",  "h8f8",  "h8g8",
    "a7a8q", "a7a8r", "a7a8b", "a7b8q", "a7b8r", "a7b8b", "b7a8q", "b7a8r",
    "b7a8b", "b7b8q", "b7b8r", "b7b8b", "b7c8q", "b7c8r", "b7c8b", "c7b8q",
    "c7b8r", "c7b8b", "c7c8q", "c7c8r", "c7c8b", "c7d8q", "c7d8r", "c7d8b",
    "d7c8q", "d7c8r", "d7c8b", "d7d8q", "d7d8r", "d7d8b", "d7e8q", "d7e8r",
    "d7e8b", "e7d8q", "e7d8r", "e7d8b", "e7e8q", "e7e8r", "e7e8b", "e7f8q",
    "e7f8r", "e7f8b", "f7e8q", "f7e8r", "f7e8b", "f7f8q", "f7f8r", "f7f8b",
    "f7g8q", "f7g8r", "f7g8b", "g7f8q", "g7f8r", "g7f8b", "g7g8q", "g7g8r",
    "g7g8b", "g7h8q", "g7h8r", "g7h8b", "h7g8q", "h7g8r", "h7g8b", "h7h8q",
    "h7h8r", "h7h8b"};
std::vector<unsigned short> BuildMoveIndices() {
  std::vector<unsigned short> res(4 * 64 * 64);
  for (size_t i = 0; i < sizeof(kIdxToMove) / sizeof(kIdxToMove[0]); ++i) {
    res[kIdxToMove[i].as_packed_int()] = i;
  }
  return res;
}
const std::vector<unsigned short> kMoveToIdx = BuildMoveIndices();
const int kKingCastleIndex =
    kMoveToIdx[BoardSquare("e1").as_int() * 64 + BoardSquare("h1").as_int()];
const int kQueenCastleIndex =
    kMoveToIdx[BoardSquare("e1").as_int() * 64 + BoardSquare("a1").as_int()];
BoardSquare Transform(BoardSquare sq, int transform) {
  if ((transform & FlipTransform) != 0) {
    sq.set(sq.row(), 7 - sq.col());
  }
  if ((transform & MirrorTransform) != 0) {
    sq.set(7 - sq.row(), sq.col());
  }
  if ((transform & TransposeTransform) != 0) {
    sq.set(7 - sq.col(), 7 - sq.row());
  }
  return sq;
}
}  // namespace
Move::Move(const std::string& str, bool black) {
  if (str.size() < 4) throw Exception("Bad move: " + str);
  SetFrom(BoardSquare(str.substr(0, 2), black));
  SetTo(BoardSquare(str.substr(2, 2), black));
  if (str.size() != 4) {
    if (str.size() != 5) throw Exception("Bad move: " + str);
    switch (str[4]) {
      case 'q':
      case 'Q':
        SetPromotion(Promotion::Queen);
        break;
      case 'r':
      case 'R':
        SetPromotion(Promotion::Rook);
        break;
      case 'b':
      case 'B':
        SetPromotion(Promotion::Bishop);
        break;
      case 'n':
      case 'N':
        SetPromotion(Promotion::Knight);
        break;
      default:
        throw Exception("Bad move: " + str);
    }
  }
}
uint16_t Move::as_packed_int() const {
  if (promotion() == Promotion::Knight) {
    return from().as_int() * 64 + to().as_int();
  } else {
    return static_cast<int>(promotion()) * 64 * 64 + from().as_int() * 64 +
           to().as_int();
  }
}
uint16_t Move::as_nn_index(int transform) const {
  if (transform == 0) {
    return kMoveToIdx[as_packed_int()];
  }
  Move transformed = *this;
  transformed.SetTo(Transform(to(), transform));
  transformed.SetFrom(Transform(from(), transform));
  return transformed.as_nn_index(0);
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/bitboard.cc

// begin of /Users/syys/CLionProjects/lc0/src/chess/board.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#if not defined(NO_PEXT)
// Include header for pext instruction.
#endif
namespace lczero {
const char* ChessBoard::kStartposFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const ChessBoard ChessBoard::kStartposBoard(ChessBoard::kStartposFen);
const BitBoard ChessBoard::kPawnMask = 0x00FFFFFFFFFFFF00ULL;
void ChessBoard::Clear() {
  std::memset(reinterpret_cast<void*>(this), 0, sizeof(ChessBoard));
}
void ChessBoard::Mirror() {
  our_pieces_.Mirror();
  their_pieces_.Mirror();
  std::swap(our_pieces_, their_pieces_);
  rooks_.Mirror();
  bishops_.Mirror();
  pawns_.Mirror();
  our_king_.Mirror();
  their_king_.Mirror();
  std::swap(our_king_, their_king_);
  castlings_.Mirror();
  flipped_ = !flipped_;
}
namespace {
static const std::pair<int, int> kKingMoves[] = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
static const std::pair<int, int> kRookDirections[] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
static const std::pair<int, int> kBishopDirections[] = {
    {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};
// Which squares can rook attack from every of squares.
static const BitBoard kRookAttacks[] = {
    0x01010101010101FEULL, 0x02020202020202FDULL, 0x04040404040404FBULL,
    0x08080808080808F7ULL, 0x10101010101010EFULL, 0x20202020202020DFULL,
    0x40404040404040BFULL, 0x808080808080807FULL, 0x010101010101FE01ULL,
    0x020202020202FD02ULL, 0x040404040404FB04ULL, 0x080808080808F708ULL,
    0x101010101010EF10ULL, 0x202020202020DF20ULL, 0x404040404040BF40ULL,
    0x8080808080807F80ULL, 0x0101010101FE0101ULL, 0x0202020202FD0202ULL,
    0x0404040404FB0404ULL, 0x0808080808F70808ULL, 0x1010101010EF1010ULL,
    0x2020202020DF2020ULL, 0x4040404040BF4040ULL, 0x80808080807F8080ULL,
    0x01010101FE010101ULL, 0x02020202FD020202ULL, 0x04040404FB040404ULL,
    0x08080808F7080808ULL, 0x10101010EF101010ULL, 0x20202020DF202020ULL,
    0x40404040BF404040ULL, 0x808080807F808080ULL, 0x010101FE01010101ULL,
    0x020202FD02020202ULL, 0x040404FB04040404ULL, 0x080808F708080808ULL,
    0x101010EF10101010ULL, 0x202020DF20202020ULL, 0x404040BF40404040ULL,
    0x8080807F80808080ULL, 0x0101FE0101010101ULL, 0x0202FD0202020202ULL,
    0x0404FB0404040404ULL, 0x0808F70808080808ULL, 0x1010EF1010101010ULL,
    0x2020DF2020202020ULL, 0x4040BF4040404040ULL, 0x80807F8080808080ULL,
    0x01FE010101010101ULL, 0x02FD020202020202ULL, 0x04FB040404040404ULL,
    0x08F7080808080808ULL, 0x10EF101010101010ULL, 0x20DF202020202020ULL,
    0x40BF404040404040ULL, 0x807F808080808080ULL, 0xFE01010101010101ULL,
    0xFD02020202020202ULL, 0xFB04040404040404ULL, 0xF708080808080808ULL,
    0xEF10101010101010ULL, 0xDF20202020202020ULL, 0xBF40404040404040ULL,
    0x7F80808080808080ULL};
// Which squares can bishop attack.
static const BitBoard kBishopAttacks[] = {
    0x8040201008040200ULL, 0x0080402010080500ULL, 0x0000804020110A00ULL,
    0x0000008041221400ULL, 0x0000000182442800ULL, 0x0000010204885000ULL,
    0x000102040810A000ULL, 0x0102040810204000ULL, 0x4020100804020002ULL,
    0x8040201008050005ULL, 0x00804020110A000AULL, 0x0000804122140014ULL,
    0x0000018244280028ULL, 0x0001020488500050ULL, 0x0102040810A000A0ULL,
    0x0204081020400040ULL, 0x2010080402000204ULL, 0x4020100805000508ULL,
    0x804020110A000A11ULL, 0x0080412214001422ULL, 0x0001824428002844ULL,
    0x0102048850005088ULL, 0x02040810A000A010ULL, 0x0408102040004020ULL,
    0x1008040200020408ULL, 0x2010080500050810ULL, 0x4020110A000A1120ULL,
    0x8041221400142241ULL, 0x0182442800284482ULL, 0x0204885000508804ULL,
    0x040810A000A01008ULL, 0x0810204000402010ULL, 0x0804020002040810ULL,
    0x1008050005081020ULL, 0x20110A000A112040ULL, 0x4122140014224180ULL,
    0x8244280028448201ULL, 0x0488500050880402ULL, 0x0810A000A0100804ULL,
    0x1020400040201008ULL, 0x0402000204081020ULL, 0x0805000508102040ULL,
    0x110A000A11204080ULL, 0x2214001422418000ULL, 0x4428002844820100ULL,
    0x8850005088040201ULL, 0x10A000A010080402ULL, 0x2040004020100804ULL,
    0x0200020408102040ULL, 0x0500050810204080ULL, 0x0A000A1120408000ULL,
    0x1400142241800000ULL, 0x2800284482010000ULL, 0x5000508804020100ULL,
    0xA000A01008040201ULL, 0x4000402010080402ULL, 0x0002040810204080ULL,
    0x0005081020408000ULL, 0x000A112040800000ULL, 0x0014224180000000ULL,
    0x0028448201000000ULL, 0x0050880402010000ULL, 0x00A0100804020100ULL,
    0x0040201008040201ULL};
// Which squares can knight attack.
static const BitBoard kKnightAttacks[] = {
    0x0000000000020400ULL, 0x0000000000050800ULL, 0x00000000000A1100ULL,
    0x0000000000142200ULL, 0x0000000000284400ULL, 0x0000000000508800ULL,
    0x0000000000A01000ULL, 0x0000000000402000ULL, 0x0000000002040004ULL,
    0x0000000005080008ULL, 0x000000000A110011ULL, 0x0000000014220022ULL,
    0x0000000028440044ULL, 0x0000000050880088ULL, 0x00000000A0100010ULL,
    0x0000000040200020ULL, 0x0000000204000402ULL, 0x0000000508000805ULL,
    0x0000000A1100110AULL, 0x0000001422002214ULL, 0x0000002844004428ULL,
    0x0000005088008850ULL, 0x000000A0100010A0ULL, 0x0000004020002040ULL,
    0x0000020400040200ULL, 0x0000050800080500ULL, 0x00000A1100110A00ULL,
    0x0000142200221400ULL, 0x0000284400442800ULL, 0x0000508800885000ULL,
    0x0000A0100010A000ULL, 0x0000402000204000ULL, 0x0002040004020000ULL,
    0x0005080008050000ULL, 0x000A1100110A0000ULL, 0x0014220022140000ULL,
    0x0028440044280000ULL, 0x0050880088500000ULL, 0x00A0100010A00000ULL,
    0x0040200020400000ULL, 0x0204000402000000ULL, 0x0508000805000000ULL,
    0x0A1100110A000000ULL, 0x1422002214000000ULL, 0x2844004428000000ULL,
    0x5088008850000000ULL, 0xA0100010A0000000ULL, 0x4020002040000000ULL,
    0x0400040200000000ULL, 0x0800080500000000ULL, 0x1100110A00000000ULL,
    0x2200221400000000ULL, 0x4400442800000000ULL, 0x8800885000000000ULL,
    0x100010A000000000ULL, 0x2000204000000000ULL, 0x0004020000000000ULL,
    0x0008050000000000ULL, 0x00110A0000000000ULL, 0x0022140000000000ULL,
    0x0044280000000000ULL, 0x0088500000000000ULL, 0x0010A00000000000ULL,
    0x0020400000000000ULL};
// Opponent pawn attacks
static const BitBoard kPawnAttacks[] = {
    0x0000000000000200ULL, 0x0000000000000500ULL, 0x0000000000000A00ULL,
    0x0000000000001400ULL, 0x0000000000002800ULL, 0x0000000000005000ULL,
    0x000000000000A000ULL, 0x0000000000004000ULL, 0x0000000000020000ULL,
    0x0000000000050000ULL, 0x00000000000A0000ULL, 0x0000000000140000ULL,
    0x0000000000280000ULL, 0x0000000000500000ULL, 0x0000000000A00000ULL,
    0x0000000000400000ULL, 0x0000000002000000ULL, 0x0000000005000000ULL,
    0x000000000A000000ULL, 0x0000000014000000ULL, 0x0000000028000000ULL,
    0x0000000050000000ULL, 0x00000000A0000000ULL, 0x0000000040000000ULL,
    0x0000000200000000ULL, 0x0000000500000000ULL, 0x0000000A00000000ULL,
    0x0000001400000000ULL, 0x0000002800000000ULL, 0x0000005000000000ULL,
    0x000000A000000000ULL, 0x0000004000000000ULL, 0x0000020000000000ULL,
    0x0000050000000000ULL, 0x00000A0000000000ULL, 0x0000140000000000ULL,
    0x0000280000000000ULL, 0x0000500000000000ULL, 0x0000A00000000000ULL,
    0x0000400000000000ULL, 0x0002000000000000ULL, 0x0005000000000000ULL,
    0x000A000000000000ULL, 0x0014000000000000ULL, 0x0028000000000000ULL,
    0x0050000000000000ULL, 0x00A0000000000000ULL, 0x0040000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL};
static const Move::Promotion kPromotions[] = {
    Move::Promotion::Queen,
    Move::Promotion::Rook,
    Move::Promotion::Bishop,
    Move::Promotion::Knight,
};
// Magic bitboard routines and structures.
// We use so-called "fancy" magic bitboards.
// Structure holding all relevant magic parameters per square.
struct MagicParams {
  // Relevant occupancy mask.
  uint64_t mask_;
  // Pointer to lookup table.
  BitBoard* attacks_table_;
#if defined(NO_PEXT)
  // Magic number.
  uint64_t magic_number_;
  // Number of bits to shift.
  uint8_t shift_bits_;
#endif
};
#if defined(NO_PEXT)
// Magic numbers determined via trial and error with random number generator
// such that the number of relevant occupancy bits suffice to index the attacks
// tables with only constructive collisions.
static const BitBoard kRookMagicNumbers[] = {
    0x088000102088C001ULL, 0x10C0200040001000ULL, 0x83001041000B2000ULL,
    0x0680280080041000ULL, 0x488004000A080080ULL, 0x0100180400010002ULL,
    0x040001C401021008ULL, 0x02000C04A980C302ULL, 0x0000800040082084ULL,
    0x5020C00820025000ULL, 0x0001002001044012ULL, 0x0402001020400A00ULL,
    0x00C0800800040080ULL, 0x4028800200040080ULL, 0x00A0804200802500ULL,
    0x8004800040802100ULL, 0x0080004000200040ULL, 0x1082810020400100ULL,
    0x0020004010080040ULL, 0x2004818010042800ULL, 0x0601010008005004ULL,
    0x4600808002001400ULL, 0x0010040009180210ULL, 0x020412000406C091ULL,
    0x040084228000C000ULL, 0x8000810100204000ULL, 0x0084110100402000ULL,
    0x0046001A00204210ULL, 0x2001040080080081ULL, 0x0144020080800400ULL,
    0x0840108400080229ULL, 0x0480308A0000410CULL, 0x0460324002800081ULL,
    0x620080A001804000ULL, 0x2800802000801006ULL, 0x0002809000800800ULL,
    0x4C09040080802800ULL, 0x4808800C00800200ULL, 0x0200311004001802ULL,
    0x0400008402002141ULL, 0x0410800140008020ULL, 0x000080C001050020ULL,
    0x004080204A020010ULL, 0x0224201001010038ULL, 0x0109001108010004ULL,
    0x0282004844020010ULL, 0x8228180110040082ULL, 0x0001000080C10002ULL,
    0x024000C120801080ULL, 0x0001406481060200ULL, 0x0101243200418600ULL,
    0x0108800800100080ULL, 0x4022080100100D00ULL, 0x0000843040600801ULL,
    0x8301000200CC0500ULL, 0x1000004500840200ULL, 0x1100104100800069ULL,
    0x2001008440001021ULL, 0x2002008830204082ULL, 0x0010145000082101ULL,
    0x01A2001004200842ULL, 0x1007000608040041ULL, 0x000A08100203028CULL,
    0x02D4048040290402ULL};
static const BitBoard kBishopMagicNumbers[] = {
    0x0008201802242020ULL, 0x0021040424806220ULL, 0x4006360602013080ULL,
    0x0004410020408002ULL, 0x2102021009001140ULL, 0x08C2021004000001ULL,
    0x6001031120200820ULL, 0x1018310402201410ULL, 0x401CE00210820484ULL,
    0x001029D001004100ULL, 0x2C00101080810032ULL, 0x0000082581000010ULL,
    0x10000A0210110020ULL, 0x200002016C202000ULL, 0x0201018821901000ULL,
    0x006A0300420A2100ULL, 0x0010014005450400ULL, 0x1008C12008028280ULL,
    0x00010010004A0040ULL, 0x3000820802044020ULL, 0x0000800405A02820ULL,
    0x8042004300420240ULL, 0x10060801210D2000ULL, 0x0210840500511061ULL,
    0x0008142118509020ULL, 0x0021109460040104ULL, 0x00A1480090019030ULL,
    0x0102008808008020ULL, 0x884084000880E001ULL, 0x040041020A030100ULL,
    0x3000810104110805ULL, 0x04040A2006808440ULL, 0x0044040404C01100ULL,
    0x4122B80800245004ULL, 0x0044020502380046ULL, 0x0100400888020200ULL,
    0x01C0002060020080ULL, 0x4008811100021001ULL, 0x8208450441040609ULL,
    0x0408004900008088ULL, 0x0294212051220882ULL, 0x000041080810E062ULL,
    0x10480A018E005000ULL, 0x80400A0204201600ULL, 0x2800200204100682ULL,
    0x0020200400204441ULL, 0x0A500600A5002400ULL, 0x801602004A010100ULL,
    0x0801841008040880ULL, 0x10010880C4200028ULL, 0x0400004424040000ULL,
    0x0401000142022100ULL, 0x00A00010020A0002ULL, 0x1010400204010810ULL,
    0x0829910400840000ULL, 0x0004235204010080ULL, 0x1002008143082000ULL,
    0x11840044440C2080ULL, 0x2802A02104030440ULL, 0x6100000900840401ULL,
    0x1C20A15A90420200ULL, 0x0088414004480280ULL, 0x0000204242881100ULL,
    0x0240080802809010ULL};
#endif
// Magic parameters for rooks/bishops.
static MagicParams rook_magic_params[64];
static MagicParams bishop_magic_params[64];
// Precomputed attacks bitboard tables.
static BitBoard rook_attacks_table[102400];
static BitBoard bishop_attacks_table[5248];
// Builds rook or bishop attacks table.
static void BuildAttacksTable(MagicParams* magic_params,
                              BitBoard* attacks_table,
                              const std::pair<int, int>* directions) {
  // Offset into lookup table.
  uint32_t table_offset = 0;
  // Initialize for all board squares.
  for (unsigned square = 0; square < 64; square++) {
    const BoardSquare b_sq(square);
    // Calculate relevant occupancy masks.
    BitBoard mask = {0};
    for (int j = 0; j < 4; j++) {
      auto direction = directions[j];
      auto dst_row = b_sq.row();
      auto dst_col = b_sq.col();
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        // If the next square in this direction is invalid, the current square
        // is at the board's edge and should not be added.
        if (!BoardSquare::IsValid(dst_row + direction.first,
                                  dst_col + direction.second))
          break;
        const BoardSquare destination(dst_row, dst_col);
        mask.set(destination);
      }
    }
    // Set mask.
    magic_params[square].mask_ = mask.as_int();
    // Cache relevant occupancy board squares.
    std::vector<BoardSquare> occupancy_squares;
    for (auto occ_sq : BitBoard(magic_params[square].mask_)) {
      occupancy_squares.emplace_back(occ_sq);
    }
#if defined(NO_PEXT)
    // Set number of shifted bits. The magic numbers have been chosen such that
    // the number of relevant occupancy bits suffice to index the attacks table.
    magic_params[square].shift_bits_ = 64 - occupancy_squares.size();
#endif
    // Set pointer to lookup table.
    magic_params[square].attacks_table_ = &attacks_table[table_offset];
    // Clear attacks table (used for sanity check later on).
    for (int i = 0; i < (1 << occupancy_squares.size()); i++) {
      attacks_table[table_offset + i] = 0;
    }
    // Build square attacks table for every possible relevant occupancy
    // bitboard.
    for (int i = 0; i < (1 << occupancy_squares.size()); i++) {
      BitBoard occupancy(0);
      for (size_t bit = 0; bit < occupancy_squares.size(); bit++) {
        occupancy.set_if(occupancy_squares[bit], (1 << bit) & i);
      }
      // Calculate attacks bitboard corresponding to this occupancy bitboard.
      BitBoard attacks(0);
      for (int j = 0; j < 4; j++) {
        auto direction = directions[j];
        auto dst_row = b_sq.row();
        auto dst_col = b_sq.col();
        while (true) {
          dst_row += direction.first;
          dst_col += direction.second;
          if (!BoardSquare::IsValid(dst_row, dst_col)) break;
          const BoardSquare destination(dst_row, dst_col);
          attacks.set(destination);
          if (occupancy.get(destination)) break;
        }
      }
#if defined(NO_PEXT)
      // Calculate magic index.
      uint64_t index = occupancy.as_int();
      index *= magic_params[square].magic_number_;
      index >>= magic_params[square].shift_bits_;
      // Sanity check. The magic numbers have been chosen such that
      // the number of relevant occupancy bits suffice to index the attacks
      // table. If the table already contains an attacks bitboard, possible
      // collisions should be constructive.
      if (attacks_table[table_offset + index] != 0 &&
          attacks_table[table_offset + index] != attacks) {
        throw Exception("Invalid magic number!");
      }
#else
      uint64_t index =
          _pext_u64(occupancy.as_int(), magic_params[square].mask_);
#endif
      // Update table.
      attacks_table[table_offset + index] = attacks;
    }
    // Update table offset.
    table_offset += (1 << occupancy_squares.size());
  }
}
// Returns the rook attacks bitboard for the given rook board square and the
// given occupied piece bitboard.
static inline BitBoard GetRookAttacks(const BoardSquare rook_square,
                                      const BitBoard pieces) {
  // Calculate magic index.
  const uint8_t square = rook_square.as_int();
#if defined(NO_PEXT)
  uint64_t index = pieces.as_int() & rook_magic_params[square].mask_;
  index *= rook_magic_params[square].magic_number_;
  index >>= rook_magic_params[square].shift_bits_;
#else
  uint64_t index = _pext_u64(pieces.as_int(), rook_magic_params[square].mask_);
#endif
  // Return attacks bitboard.
  return rook_magic_params[square].attacks_table_[index];
}
// Returns the bishop attacks bitboard for the given bishop board square and
// the given occupied piece bitboard.
static inline BitBoard GetBishopAttacks(const BoardSquare bishop_square,
                                        const BitBoard pieces) {
  // Calculate magic index.
  const uint8_t square = bishop_square.as_int();
#if defined(NO_PEXT)
  uint64_t index = pieces.as_int() & bishop_magic_params[square].mask_;
  index *= bishop_magic_params[square].magic_number_;
  index >>= bishop_magic_params[square].shift_bits_;
#else
  uint64_t index =
      _pext_u64(pieces.as_int(), bishop_magic_params[square].mask_);
#endif
  // Return attacks bitboard.
  return bishop_magic_params[square].attacks_table_[index];
}
}  // namespace
void InitializeMagicBitboards() {
#if defined(NO_PEXT)
  // Set magic numbers for all board squares.
  for (unsigned square = 0; square < 64; square++) {
    rook_magic_params[square].magic_number_ =
        kRookMagicNumbers[square].as_int();
    bishop_magic_params[square].magic_number_ =
        kBishopMagicNumbers[square].as_int();
  }
#endif
  // Build attacks tables.
  BuildAttacksTable(rook_magic_params, rook_attacks_table, kRookDirections);
  BuildAttacksTable(bishop_magic_params, bishop_attacks_table,
                    kBishopDirections);
}
MoveList ChessBoard::GeneratePseudolegalMoves() const {
  MoveList result;
  result.reserve(60);
  for (auto source : our_pieces_) {
    // King
    if (source == our_king_) {
      for (const auto& delta : kKingMoves) {
        const auto dst_row = source.row() + delta.first;
        const auto dst_col = source.col() + delta.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) continue;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) continue;
        if (IsUnderAttack(destination)) continue;
        result.emplace_back(source, destination);
      }
      // Castlings.
      auto walk_free = [this](int from, int to, int rook, int king) {
        for (int i = from; i <= to; ++i) {
          if (i == rook || i == king) continue;
          if (our_pieces_.get(i) || their_pieces_.get(i)) return false;
        }
        return true;
      };
      // @From may be less or greater than @to. @To is not included in check
      // unless it is the same with @from.
      auto range_attacked = [this](int from, int to) {
        if (from == to) return IsUnderAttack(from);
        const int increment = from < to ? 1 : -1;
        while (from != to) {
          if (IsUnderAttack(from)) return true;
          from += increment;
        }
        return false;
      };
      const uint8_t king = source.col();
      // For castlings we don't check destination king square for checks, it
      // will be done in legal move check phase.
      if (castlings_.we_can_000()) {
        const uint8_t qrook = castlings_.queenside_rook();
        if (walk_free(std::min(static_cast<uint8_t>(C1), qrook),
                      std::max(static_cast<uint8_t>(D1), king), qrook, king) &&
            !range_attacked(king, C1)) {
          result.emplace_back(source,
                              BoardSquare(RANK_1, castlings_.queenside_rook()));
        }
      }
      if (castlings_.we_can_00()) {
        const uint8_t krook = castlings_.kingside_rook();
        if (walk_free(std::min(static_cast<uint8_t>(F1), king),
                      std::max(static_cast<uint8_t>(G1), krook), krook, king) &&
            !range_attacked(king, G1)) {
          result.emplace_back(source,
                              BoardSquare(RANK_1, castlings_.kingside_rook()));
        }
      }
      continue;
    }
    bool processed_piece = false;
    // Rook (and queen)
    if (rooks_.get(source)) {
      processed_piece = true;
      BitBoard attacked =
          GetRookAttacks(source, our_pieces_ | their_pieces_) - our_pieces_;
      for (const auto& destination : attacked) {
        result.emplace_back(source, destination);
      }
    }
    // Bishop (and queen)
    if (bishops_.get(source)) {
      processed_piece = true;
      BitBoard attacked =
          GetBishopAttacks(source, our_pieces_ | their_pieces_) - our_pieces_;
      for (const auto& destination : attacked) {
        result.emplace_back(source, destination);
      }
    }
    if (processed_piece) continue;
    // Pawns.
    if ((pawns_ & kPawnMask).get(source)) {
      // Moves forward.
      {
        const auto dst_row = source.row() + 1;
        const auto dst_col = source.col();
        const BoardSquare destination(dst_row, dst_col);
        if (!our_pieces_.get(destination) && !their_pieces_.get(destination)) {
          if (dst_row != RANK_8) {
            result.emplace_back(source, destination);
            if (dst_row == RANK_3) {
              // Maybe it'll be possible to move two squares.
              if (!our_pieces_.get(RANK_4, dst_col) &&
                  !their_pieces_.get(RANK_4, dst_col)) {
                result.emplace_back(source, BoardSquare(RANK_4, dst_col));
              }
            }
          } else {
            // Promotions
            for (auto promotion : kPromotions) {
              result.emplace_back(source, destination, promotion);
            }
          }
        }
      }
      // Captures.
      {
        for (auto direction : {-1, 1}) {
          const auto dst_row = source.row() + 1;
          const auto dst_col = source.col() + direction;
          if (dst_col < 0 || dst_col >= 8) continue;
          const BoardSquare destination(dst_row, dst_col);
          if (their_pieces_.get(destination)) {
            if (dst_row == RANK_8) {
              // Promotion.
              for (auto promotion : kPromotions) {
                result.emplace_back(source, destination, promotion);
              }
            } else {
              // Ordinary capture.
              result.emplace_back(source, destination);
            }
          } else if (dst_row == RANK_6 && pawns_.get(RANK_8, dst_col)) {
            // En passant.
            // "Pawn" on opponent's file 8 means that en passant is possible.
            // Those fake pawns are reset in ApplyMove.
            result.emplace_back(source, destination);
          }
        }
      }
      continue;
    }
    // Knight.
    {
      for (const auto destination :
           kKnightAttacks[source.as_int()] - our_pieces_) {
        result.emplace_back(source, destination);
      }
    }
  }
  return result;
}  // namespace lczero
bool ChessBoard::ApplyMove(Move move) {
  const auto& from = move.from();
  const auto& to = move.to();
  const auto from_row = from.row();
  const auto from_col = from.col();
  const auto to_row = to.row();
  const auto to_col = to.col();
  // Castlings.
  if (from == our_king_) {
    castlings_.reset_we_can_00();
    castlings_.reset_we_can_000();
    auto do_castling = [this](int king_dst, int rook_src, int rook_dst) {
      // Remove en passant flags.
      pawns_ &= kPawnMask;
      our_pieces_.reset(our_king_);
      our_pieces_.reset(rook_src);
      rooks_.reset(rook_src);
      our_pieces_.set(king_dst);
      our_pieces_.set(rook_dst);
      rooks_.set(rook_dst);
      our_king_ = king_dst;
    };
    if (from_row == RANK_1 && to_row == RANK_1) {
      const auto our_rooks = rooks() & our_pieces_;
      if (our_rooks.get(to)) {
        // Castling.
        if (to_col > from_col) {
          // Kingside.
          do_castling(G1, to.as_int(), F1);
        } else {
          // Queenside.
          do_castling(C1, to.as_int(), D1);
        }
        return false;
      } else if (from_col == FILE_E && to_col == FILE_G) {
        // Non FRC-style e1g1 castling (as opposed to e1h1).
        do_castling(G1, H1, F1);
        return false;
      } else if (from_col == FILE_E && to_col == FILE_C) {
        // Non FRC-style e1c1 castling (as opposed to e1a1).
        do_castling(C1, A1, D1);
        return false;
      }
    }
  }
  // Move in our pieces.
  our_pieces_.reset(from);
  our_pieces_.set(to);
  // Remove captured piece.
  bool reset_50_moves = their_pieces_.get(to);
  their_pieces_.reset(to);
  rooks_.reset(to);
  bishops_.reset(to);
  pawns_.reset(to);
  if (to.as_int() == 56 + castlings_.kingside_rook()) {
    castlings_.reset_they_can_00();
  }
  if (to.as_int() == 56 + castlings_.queenside_rook()) {
    castlings_.reset_they_can_000();
  }
  // En passant.
  if (from_row == RANK_5 && pawns_.get(from) && from_col != to_col &&
      pawns_.get(RANK_8, to_col)) {
    pawns_.reset(RANK_5, to_col);
    their_pieces_.reset(RANK_5, to_col);
  }
  // Remove en passant flags.
  pawns_ &= kPawnMask;
  // If pawn was moved, reset 50 move draw counter.
  reset_50_moves |= pawns_.get(from);
  // King, non-castling move
  if (from == our_king_) {
    our_king_ = to;
    return reset_50_moves;
  }
  // Promotion.
  if (to_row == RANK_8 && pawns_.get(from)) {
    switch (move.promotion()) {
      case Move::Promotion::Rook:
        rooks_.set(to);
        break;
      case Move::Promotion::Bishop:
        bishops_.set(to);
        break;
      case Move::Promotion::Queen:
        rooks_.set(to);
        bishops_.set(to);
        break;
      default:;
    }
    pawns_.reset(from);
    return true;
  }
  // Reset castling rights.
  if (from_row == RANK_1 && rooks_.get(from)) {
    if (from_col == castlings_.queenside_rook()) castlings_.reset_we_can_000();
    if (from_col == castlings_.kingside_rook()) castlings_.reset_we_can_00();
  }
  // Ordinary move.
  rooks_.set_if(to, rooks_.get(from));
  bishops_.set_if(to, bishops_.get(from));
  pawns_.set_if(to, pawns_.get(from));
  rooks_.reset(from);
  bishops_.reset(from);
  pawns_.reset(from);
  // Set en passant flag.
  if (to_row - from_row == 2 && pawns_.get(to)) {
    BoardSquare ep_sq(to_row - 1, to_col);
    if (kPawnAttacks[ep_sq.as_int()].intersects(their_pieces_ & pawns_)) {
      pawns_.set(0, to_col);
    }
  }
  return reset_50_moves;
}
bool ChessBoard::IsUnderAttack(BoardSquare square) const {
  const int row = square.row();
  const int col = square.col();
  // Check king.
  {
    const int krow = their_king_.row();
    const int kcol = their_king_.col();
    if (std::abs(krow - row) <= 1 && std::abs(kcol - col) <= 1) return true;
  }
  // Check rooks (and queens).
  if (GetRookAttacks(square, our_pieces_ | their_pieces_)
          .intersects(their_pieces_ & rooks_)) {
    return true;
  }
  // Check bishops.
  if (GetBishopAttacks(square, our_pieces_ | their_pieces_)
          .intersects(their_pieces_ & bishops_)) {
    return true;
  }
  // Check pawns.
  if (kPawnAttacks[square.as_int()].intersects(their_pieces_ & pawns_)) {
    return true;
  }
  // Check knights.
  {
    if (kKnightAttacks[square.as_int()].intersects(their_pieces_ - their_king_ -
                                                   rooks_ - bishops_ -
                                                   (pawns_ & kPawnMask))) {
      return true;
    }
  }
  return false;
}
bool ChessBoard::IsSameMove(Move move1, Move move2) const {
  // If moves are equal, it's the same move.
  if (move1 == move2) return true;
  // Explicitly check all legacy castling moves. Need to check for king, for
  // e.g. rook e1a1 and e1c1 are different moves.
  if (move1.from() != move2.from() || move1.from() != E1 ||
      our_king_ != move1.from()) {
    return false;
  }
  if (move1.to() == A1 && move2.to() == C1) return true;
  if (move1.to() == C1 && move2.to() == A1) return true;
  if (move1.to() == G1 && move2.to() == H1) return true;
  if (move1.to() == H1 && move2.to() == G1) return true;
  return false;
}
Move ChessBoard::GetLegacyMove(Move move) const {
  if (our_king_ != move.from() || !our_pieces_.get(move.to())) {
    return move;
  }
  if (move == Move(E1, H1)) return Move(E1, G1);
  if (move == Move(E1, A1)) return Move(E1, C1);
  return move;
}
Move ChessBoard::GetModernMove(Move move) const {
  if (our_king_ != E1 || move.from() != E1) return move;
  if (move == Move(E1, G1) && !our_pieces_.get(G1)) return Move(E1, H1);
  if (move == Move(E1, C1) && !our_pieces_.get(C1)) return Move(E1, A1);
  return move;
}
KingAttackInfo ChessBoard::GenerateKingAttackInfo() const {
  KingAttackInfo king_attack_info;
  // Number of attackers that give check (used for double check detection).
  unsigned num_king_attackers = 0;
  const int row = our_king_.row();
  const int col = our_king_.col();
  // King checks are unnecessary, as kings cannot give check.
  // Check rooks (and queens).
  if (kRookAttacks[our_king_.as_int()].intersects(their_pieces_ & rooks_)) {
    for (const auto& direction : kRookDirections) {
      auto dst_row = row;
      auto dst_col = col;
      BitBoard attack_line(0);
      bool possible_pinned_piece_found = false;
      BoardSquare possible_pinned_piece;
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) break;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) {
          if (possible_pinned_piece_found) {
            // No pieces pinned.
            break;
          } else {
            // This is a possible pinned piece.
            possible_pinned_piece_found = true;
            possible_pinned_piece = destination;
          }
        }
        if (!possible_pinned_piece_found) {
          attack_line.set(destination);
        }
        if (their_pieces_.get(destination)) {
          if (rooks_.get(destination)) {
            if (possible_pinned_piece_found) {
              // Store the pinned piece.
              king_attack_info.pinned_pieces_.set(possible_pinned_piece);
            } else {
              // Update attack lines.
              king_attack_info.attack_lines_ =
                  king_attack_info.attack_lines_ | attack_line;
              num_king_attackers++;
            }
          }
          break;
        }
      }
    }
  }
  // Check bishops.
  if (kBishopAttacks[our_king_.as_int()].intersects(their_pieces_ & bishops_)) {
    for (const auto& direction : kBishopDirections) {
      auto dst_row = row;
      auto dst_col = col;
      BitBoard attack_line(0);
      bool possible_pinned_piece_found = false;
      BoardSquare possible_pinned_piece;
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) break;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) {
          if (possible_pinned_piece_found) {
            // No pieces pinned.
            break;
          } else {
            // This is a possible pinned piece.
            possible_pinned_piece_found = true;
            possible_pinned_piece = destination;
          }
        }
        if (!possible_pinned_piece_found) {
          attack_line.set(destination);
        }
        if (their_pieces_.get(destination)) {
          if (bishops_.get(destination)) {
            if (possible_pinned_piece_found) {
              // Store the pinned piece.
              king_attack_info.pinned_pieces_.set(possible_pinned_piece);
            } else {
              // Update attack lines.
              king_attack_info.attack_lines_ =
                  king_attack_info.attack_lines_ | attack_line;
              num_king_attackers++;
            }
          }
          break;
        }
      }
    }
  }
  // Check pawns.
  const BitBoard attacking_pawns =
      kPawnAttacks[our_king_.as_int()] & their_pieces_ & pawns_;
  king_attack_info.attack_lines_ =
      king_attack_info.attack_lines_ | attacking_pawns;
  if (attacking_pawns.as_int()) {
    // No more than one pawn can give check.
    num_king_attackers++;
  }
  // Check knights.
  const BitBoard attacking_knights =
      kKnightAttacks[our_king_.as_int()] &
      (their_pieces_ - their_king_ - rooks_ - bishops_ - (pawns_ & kPawnMask));
  king_attack_info.attack_lines_ =
      king_attack_info.attack_lines_ | attacking_knights;
  if (attacking_knights.as_int()) {
    // No more than one knight can give check.
    num_king_attackers++;
  }
  assert(num_king_attackers <= 2);
  king_attack_info.double_check_ = (num_king_attackers == 2);
  return king_attack_info;
}
bool ChessBoard::IsLegalMove(Move move,
                             const KingAttackInfo& king_attack_info) const {
  const auto& from = move.from();
  const auto& to = move.to();
  // En passant. Complex but rare. Just apply
  // and check that we are not under check.
  if (from.row() == 4 && pawns_.get(from) && from.col() != to.col() &&
      pawns_.get(7, to.col())) {
    ChessBoard board(*this);
    board.ApplyMove(move);
    return !board.IsUnderCheck();
  }
  // Check if we are already under check.
  if (king_attack_info.in_check()) {
    // King move.
    if (from == our_king_) {
      // Just apply and check that we are not under check.
      ChessBoard board(*this);
      board.ApplyMove(move);
      return !board.IsUnderCheck();
    }
    // Pinned pieces can never resolve a check.
    if (king_attack_info.is_pinned(from)) {
      return false;
    }
    // The piece to move is no king and is not pinned.
    if (king_attack_info.in_double_check()) {
      // Only a king move can resolve the double check.
      return false;
    } else {
      // Only one attacking piece gives check.
      // Our piece is free to move (not pinned). Check if the attacker is
      // captured or interposed after the piece has moved to its destination
      // square.
      return king_attack_info.is_on_attack_line(to);
    }
  }
  // King moves.
  if (from == our_king_) {
    if (from.row() != 0 || to.row() != 0 ||
        (abs(from.col() - to.col()) == 1 && !our_pieces_.get(to))) {
      // Non-castling move. Already checked during movegen.
      return true;
    }
    // Checking whether king is under check after castling.
    ChessBoard board(*this);
    board.ApplyMove(move);
    return !board.IsUnderCheck();
  }
  // If we get here, we are not under check.
  // If the piece is not pinned, it is free to move anywhere.
  if (!king_attack_info.is_pinned(from)) return true;
  // The piece is pinned. Now check that it stays on the same line w.r.t. the
  // king.
  const int dx_from = from.col() - our_king_.col();
  const int dy_from = from.row() - our_king_.row();
  const int dx_to = to.col() - our_king_.col();
  const int dy_to = to.row() - our_king_.row();
  if (dx_from == 0 || dx_to == 0) {
    return (dx_from == dx_to);
  } else {
    return (dx_from * dy_to == dx_to * dy_from);
  }
}
MoveList ChessBoard::GenerateLegalMoves() const {
  const KingAttackInfo king_attack_info = GenerateKingAttackInfo();
  MoveList result = GeneratePseudolegalMoves();
  result.erase(
      std::remove_if(result.begin(), result.end(),
                     [&](Move m) { return !IsLegalMove(m, king_attack_info); }),
      result.end());
  return result;
}
void ChessBoard::SetFromFen(std::string fen, int* rule50_ply, int* moves) {
  Clear();
  int row = 7;
  int col = 0;
  // Remove any trailing whitespaces to detect eof after the last field.
  fen.erase(std::find_if(fen.rbegin(), fen.rend(),
                         [](char c) { return !std::isspace(c); })
                .base(),
            fen.end());
  std::istringstream fen_str(fen);
  std::string board;
  fen_str >> board;
  std::string who_to_move = "w";
  if (!fen_str.eof()) fen_str >> who_to_move;
  // Assume no castling rights. Other engines, e.g., Stockfish, assume kings and
  // rooks on their initial rows can each castle with the outer-most rook.  Our
  // implementation currently supports 960 castling where white and black rooks
  // have matching columns, so it's unclear which rights to assume.
  std::string castlings = "-";
  if (!fen_str.eof()) fen_str >> castlings;
  std::string en_passant = "-";
  if (!fen_str.eof()) fen_str >> en_passant;
  int rule50_halfmoves = 0;
  if (!fen_str.eof()) fen_str >> rule50_halfmoves;
  int total_moves = 1;
  if (!fen_str.eof()) fen_str >> total_moves;
  if (!fen_str) throw Exception("Bad fen string: " + fen);
  for (char c : board) {
    if (c == '/') {
      --row;
      if (row < 0) throw Exception("Bad fen string (too many rows): " + fen);
      col = 0;
      continue;
    }
    if (std::isdigit(c)) {
      col += c - '0';
      continue;
    }
    if (col >= 8) throw Exception("Bad fen string (too many columns): " + fen);
    if (std::isupper(c)) {
      // White piece.
      our_pieces_.set(row, col);
    } else {
      // Black piece.
      their_pieces_.set(row, col);
    }
    if (c == 'K') {
      our_king_.set(row, col);
    } else if (c == 'k') {
      their_king_.set(row, col);
    } else if (c == 'R' || c == 'r') {
      rooks_.set(row, col);
    } else if (c == 'B' || c == 'b') {
      bishops_.set(row, col);
    } else if (c == 'Q' || c == 'q') {
      rooks_.set(row, col);
      bishops_.set(row, col);
    } else if (c == 'P' || c == 'p') {
      if (row == 7 || row == 0) {
        throw Exception("Bad fen string (pawn in first/last row): " + fen);
      }
      pawns_.set(row, col);
    } else if (c == 'N' || c == 'n') {
      // Do nothing
    } else {
      throw Exception("Bad fen string: " + fen);
    }
    ++col;
  }
  if (castlings != "-") {
    uint8_t left_rook = FILE_A;
    uint8_t right_rook = FILE_H;
    for (char c : castlings) {
      const bool is_black = std::islower(c);
      const int king_col = (is_black ? their_king_ : our_king_).col();
      if (!is_black) c = std::tolower(c);
      const auto rooks =
          (is_black ? their_pieces_ : our_pieces_) & ChessBoard::rooks();
      if (c == 'k') {
        // Finding rightmost rook.
        for (right_rook = FILE_H; right_rook > king_col; --right_rook) {
          if (rooks.get(is_black ? RANK_8 : RANK_1, right_rook)) break;
        }
        if (right_rook == king_col) {
          throw Exception("Bad fen string (no kingside rook): " + fen);
        }
        if (is_black) {
          castlings_.set_they_can_00();
        } else {
          castlings_.set_we_can_00();
        }
      } else if (c == 'q') {
        // Finding leftmost rook.
        for (left_rook = FILE_A; left_rook < king_col; ++left_rook) {
          if (rooks.get(is_black ? RANK_8 : RANK_1, left_rook)) break;
        }
        if (left_rook == king_col) {
          throw Exception("Bad fen string (no queenside rook): " + fen);
        }
        if (is_black) {
          castlings_.set_they_can_000();
        } else {
          castlings_.set_we_can_000();
        }
      } else if (c >= 'a' && c <= 'h') {
        int rook_col = c - 'a';
        if (rook_col < king_col) {
          left_rook = rook_col;
          if (is_black) {
            castlings_.set_they_can_000();
          } else {
            castlings_.set_we_can_000();
          }
        } else {
          right_rook = rook_col;
          if (is_black) {
            castlings_.set_they_can_00();
          } else {
            castlings_.set_we_can_00();
          }
        }
      } else {
        throw Exception("Bad fen string (unexpected casting symbol): " + fen);
      }
    }
    castlings_.SetRookPositions(left_rook, right_rook);
  }
  if (en_passant != "-") {
    auto square = BoardSquare(en_passant);
    if (square.row() != RANK_3 && square.row() != RANK_6)
      throw Exception("Bad fen string: " + fen + " wrong en passant rank");
    pawns_.set((square.row() == RANK_3) ? RANK_1 : RANK_8, square.col());
  }
  if (who_to_move == "b" || who_to_move == "B") {
    Mirror();
  } else if (who_to_move != "w" && who_to_move != "W") {
    throw Exception("Bad fen string (side to move): " + fen);
  }
  if (rule50_ply) *rule50_ply = rule50_halfmoves;
  if (moves) *moves = total_moves;
}
bool ChessBoard::HasMatingMaterial() const {
  if (!rooks_.empty() || !pawns_.empty()) {
    return true;
  }
  if ((our_pieces_ | their_pieces_).count() < 4) {
    // K v K, K+B v K, K+N v K.
    return false;
  }
  if (!(knights().empty())) {
    return true;
  }
  // Only kings and bishops remain.
  constexpr BitBoard kLightSquares(0x55AA55AA55AA55AAULL);
  constexpr BitBoard kDarkSquares(0xAA55AA55AA55AA55ULL);
  const bool light_bishop = bishops_.intersects(kLightSquares);
  const bool dark_bishop = bishops_.intersects(kDarkSquares);
  return light_bishop && dark_bishop;
}
std::string ChessBoard::DebugString() const {
  std::string result;
  for (int i = 7; i >= 0; --i) {
    for (int j = 0; j < 8; ++j) {
      if (!our_pieces_.get(i, j) && !their_pieces_.get(i, j)) {
        if (i == 2 && pawns_.get(0, j))
          result += '*';
        else if (i == 5 && pawns_.get(7, j))
          result += '*';
        else
          result += '.';
        continue;
      }
      if (our_king_ == i * 8 + j) {
        result += 'K';
        continue;
      }
      if (their_king_ == i * 8 + j) {
        result += 'k';
        continue;
      }
      char c = '?';
      if ((pawns_ & kPawnMask).get(i, j)) {
        c = 'p';
      } else if (bishops_.get(i, j)) {
        if (rooks_.get(i, j))
          c = 'q';
        else
          c = 'b';
      } else if (rooks_.get(i, j)) {
        c = 'r';
      } else {
        c = 'n';
      }
      if (our_pieces_.get(i, j)) c = std::toupper(c);
      result += c;
    }
    if (i == 0) {
      result += " " + castlings_.DebugString();
      result += flipped_ ? " (from black's eyes)" : " (from white's eyes)";
      result += " Hash: " + std::to_string(Hash());
    }
    result += '\n';
  }
  return result;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/board.cc

// begin of /Users/syys/CLionProjects/lc0/src/chess/position.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace {
// GetPieceAt returns the piece found at row, col on board or the null-char '\0'
// in case no piece there.
char GetPieceAt(const lczero::ChessBoard& board, int row, int col) {
  char c = '\0';
  if (board.ours().get(row, col) || board.theirs().get(row, col)) {
    if (board.pawns().get(row, col)) {
      c = 'P';
    } else if (board.kings().get(row, col)) {
      c = 'K';
    } else if (board.bishops().get(row, col)) {
      c = 'B';
    } else if (board.queens().get(row, col)) {
      c = 'Q';
    } else if (board.rooks().get(row, col)) {
      c = 'R';
    } else {
      c = 'N';
    }
    if (board.theirs().get(row, col)) {
      c = std::tolower(c);  // Capitals are for white.
    }
  }
  return c;
}
}  // namespace
namespace lczero {
Position::Position(const Position& parent, Move m)
    : rule50_ply_(parent.rule50_ply_ + 1), ply_count_(parent.ply_count_ + 1) {
  them_board_ = parent.us_board_;
  const bool is_zeroing = them_board_.ApplyMove(m);
  us_board_ = them_board_;
  us_board_.Mirror();
  if (is_zeroing) rule50_ply_ = 0;
}
Position::Position(const ChessBoard& board, int rule50_ply, int game_ply)
    : rule50_ply_(rule50_ply), repetitions_(0), ply_count_(game_ply) {
  us_board_ = board;
  them_board_ = board;
  them_board_.Mirror();
}
uint64_t Position::Hash() const {
  return HashCat({us_board_.Hash(), static_cast<unsigned long>(repetitions_)});
}
std::string Position::DebugString() const { return us_board_.DebugString(); }
GameResult operator-(const GameResult& res) {
  return res == GameResult::BLACK_WON   ? GameResult::WHITE_WON
         : res == GameResult::WHITE_WON ? GameResult::BLACK_WON
                                        : res;
}
GameResult PositionHistory::ComputeGameResult() const {
  const auto& board = Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  if (legal_moves.empty()) {
    if (board.IsUnderCheck()) {
      // Checkmate.
      return IsBlackToMove() ? GameResult::WHITE_WON : GameResult::BLACK_WON;
    }
    // Stalemate.
    return GameResult::DRAW;
  }
  if (!board.HasMatingMaterial()) return GameResult::DRAW;
  if (Last().GetRule50Ply() >= 100) return GameResult::DRAW;
  if (Last().GetRepetitions() >= 2) return GameResult::DRAW;
  return GameResult::UNDECIDED;
}
void PositionHistory::Reset(const ChessBoard& board, int rule50_ply,
                            int game_ply) {
  positions_.clear();
  positions_.emplace_back(board, rule50_ply, game_ply);
}
void PositionHistory::Append(Move m) {
  // TODO(mooskagh) That should be emplace_back(Last(), m), but MSVS STL
  //                has a bug in implementation of emplace_back, when
  //                reallocation happens. (it also reallocates Last())
  positions_.push_back(Position(Last(), m));
  int cycle_length;
  int repetitions = ComputeLastMoveRepetitions(&cycle_length);
  positions_.back().SetRepetitions(repetitions, cycle_length);
}
int PositionHistory::ComputeLastMoveRepetitions(int* cycle_length) const {
  *cycle_length = 0;
  const auto& last = positions_.back();
  // TODO(crem) implement hash/cache based solution.
  if (last.GetRule50Ply() < 4) return 0;
  for (int idx = positions_.size() - 3; idx >= 0; idx -= 2) {
    const auto& pos = positions_[idx];
    if (pos.GetBoard() == last.GetBoard()) {
      *cycle_length = positions_.size() - 1 - idx;
      return 1 + pos.GetRepetitions();
    }
    if (pos.GetRule50Ply() < 2) return 0;
  }
  return 0;
}
bool PositionHistory::DidRepeatSinceLastZeroingMove() const {
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (iter->GetRepetitions() > 0) return true;
    if (iter->GetRule50Ply() == 0) return false;
  }
  return false;
}
uint64_t PositionHistory::HashLast(int positions) const {
  uint64_t hash = positions;
  for (auto iter = positions_.rbegin(), end = positions_.rend(); iter != end;
       ++iter) {
    if (!positions--) break;
    hash = HashCat(hash, iter->Hash());
  }
  return HashCat(hash, Last().GetRule50Ply());
}
std::string GetFen(const Position& pos) {
  std::string result;
  const ChessBoard& board = pos.GetWhiteBoard();
  for (int row = 7; row >= 0; --row) {
    int emptycounter = 0;
    for (int col = 0; col < 8; ++col) {
      char piece = GetPieceAt(board, row, col);
      if (emptycounter > 0 && piece) {
        result += std::to_string(emptycounter);
        emptycounter = 0;
      }
      if (piece) {
        result += piece;
      } else {
        emptycounter++;
      }
    }
    if (emptycounter > 0) result += std::to_string(emptycounter);
    if (row > 0) result += "/";
  }
  std::string enpassant = "-";
  if (!board.en_passant().empty()) {
    auto sq = *board.en_passant().begin();
    enpassant = BoardSquare(pos.IsBlackToMove() ? 2 : 5, sq.col()).as_string();
  }
  result += pos.IsBlackToMove() ? " b" : " w";
  result += " " + board.castlings().as_string();
  result += " " + enpassant;
  result += " " + std::to_string(pos.GetRule50Ply());
  result += " " + std::to_string(
                      (pos.GetGamePly() + (pos.IsBlackToMove() ? 1 : 2)) / 2);
  return result;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/position.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/logging.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const size_t kBufferSizeLines = 200;
const char* const kStderrFilename = "<stderr>";
}  // namespace
Logging& Logging::Get() {
  static Logging logging;
  return logging;
}
void Logging::WriteLineRaw(const std::string& line) {
  Mutex::Lock lock_(mutex_);
  if (filename_.empty()) {
    buffer_.push_back(line);
    if (buffer_.size() > kBufferSizeLines) buffer_.pop_front();
  } else {
    auto& file = (filename_ == kStderrFilename) ? std::cerr : file_;
    file << line << std::endl;
  }
}
void Logging::SetFilename(const std::string& filename) {
  Mutex::Lock lock_(mutex_);
  if (filename_ == filename) return;
  filename_ = filename;
  if (filename.empty() || filename == kStderrFilename) {
    file_.close();
  }
  if (filename.empty()) return;
  if (filename != kStderrFilename) file_.open(filename, std::ios_base::app);
  auto& file = (filename == kStderrFilename) ? std::cerr : file_;
  file << "\n\n============= Log started. =============" << std::endl;
  for (const auto& line : buffer_) file << line << std::endl;
  buffer_.clear();
}
LogMessage::LogMessage(const char* file, int line) {
  *this << FormatTime(std::chrono::system_clock::now()) << ' '
        << std::setfill(' ') << std::this_thread::get_id() << std::setfill('0')
        << ' ' << file << ':' << line << "] ";
}
LogMessage::~LogMessage() { Logging::Get().WriteLineRaw(str()); }
StderrLogMessage::StderrLogMessage(const char* file, int line)
    : log_(file, line) {}
StderrLogMessage::~StderrLogMessage() {
  std::cerr << str() << std::endl;
  log_ << str();
}
StdoutLogMessage::StdoutLogMessage(const char* file, int line)
    : log_(file, line) {}
StdoutLogMessage::~StdoutLogMessage() {
  std::cout << str() << std::endl;
  log_ << str();
}
std::chrono::time_point<std::chrono::system_clock> SteadyClockToSystemClock(
    std::chrono::time_point<std::chrono::steady_clock> time) {
  return std::chrono::system_clock::now() +
         std::chrono::duration_cast<std::chrono::system_clock::duration>(
             time - std::chrono::steady_clock::now());
}
std::string FormatTime(
    std::chrono::time_point<std::chrono::system_clock> time) {
  static Mutex mutex;
  std::ostringstream ss;
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  const auto us =
      duration_cast<microseconds>(time.time_since_epoch()).count() % 1000000;
  auto timer = std::chrono::system_clock::to_time_t(time);
  // std::localtime is not thread safe. Since this is the only place
  // std::localtime is used in the program, guard by mutex.
  // TODO: replace with std::localtime_r or s once they are properly
  // standardised. Or some other more c++ like time component thing, whichever
  // comes first...
  {
    Mutex::Lock lock(mutex);
    ss << std::put_time(std::localtime(&timer), "%m%d %T") << '.'
       << std::setfill('0') << std::setw(6) << us;
  }
  return ss.str();
}
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/utils/logging.cc

// begin of /Users/syys/CLionProjects/lc0/src/chess/uciloop.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const std::unordered_map<std::string, std::unordered_set<std::string>>
    kKnownCommands = {
        {{"uci"}, {}},
        {{"isready"}, {}},
        {{"setoption"}, {"context", "name", "value"}},
        {{"ucinewgame"}, {}},
        {{"position"}, {"fen", "startpos", "moves"}},
        {{"go"},
         {"infinite", "wtime", "btime", "winc", "binc", "movestogo", "depth",
          "nodes", "movetime", "searchmoves", "ponder"}},
        {{"start"}, {}},
        {{"stop"}, {}},
        {{"ponderhit"}, {}},
        {{"quit"}, {}},
        {{"xyzzy"}, {}},
        {{"fen"}, {}},
};
std::pair<std::string, std::unordered_map<std::string, std::string>>
ParseCommand(const std::string& line) {
  std::unordered_map<std::string, std::string> params;
  std::string* value = nullptr;
  std::istringstream iss(line);
  std::string token;
  iss >> token >> std::ws;
  // If empty line, return empty command.
  if (token.empty()) return {};
  const auto command = kKnownCommands.find(token);
  if (command == kKnownCommands.end()) {
    throw Exception("Unknown command: " + line);
  }
  std::string whitespace;
  while (iss >> token) {
    auto iter = command->second.find(token);
    if (iter == command->second.end()) {
      if (!value) throw Exception("Unexpected token: " + token);
      *value += whitespace + token;
      whitespace = " ";
    } else {
      value = &params[token];
      iss >> std::ws;
      whitespace = "";
    }
  }
  return {command->first, params};
}
std::string GetOrEmpty(
    const std::unordered_map<std::string, std::string>& params,
    const std::string& key) {
  const auto iter = params.find(key);
  if (iter == params.end()) return {};
  return iter->second;
}
int GetNumeric(const std::unordered_map<std::string, std::string>& params,
               const std::string& key) {
  const auto iter = params.find(key);
  if (iter == params.end()) {
    throw Exception("Unexpected error");
  }
  const std::string& str = iter->second;
  try {
    if (str.empty()) {
      throw Exception("expected value after " + key);
    }
    return std::stoi(str);
  } catch (std::invalid_argument&) {
    throw Exception("invalid value " + str);
  } catch (const std::out_of_range&) {
    throw Exception("out of range value " + str);
  }
}
bool ContainsKey(const std::unordered_map<std::string, std::string>& params,
                 const std::string& key) {
  return params.find(key) != params.end();
}
}  // namespace
void UciLoop::RunLoop() {
  std::cout.setf(std::ios::unitbuf);
  std::string line;
  while (std::getline(std::cin, line)) {
    LOGFILE << ">> " << line;
    try {
      auto command = ParseCommand(line);
      // Ignore empty line.
      if (command.first.empty()) continue;
      if (!DispatchCommand(command.first, command.second)) break;
    } catch (Exception& ex) {
      SendResponse(std::string("error ") + ex.what());
    }
  }
}
bool UciLoop::DispatchCommand(
    const std::string& command,
    const std::unordered_map<std::string, std::string>& params) {
  if (command == "uci") {
    CmdUci();
  } else if (command == "isready") {
    CmdIsReady();
  } else if (command == "setoption") {
    CmdSetOption(GetOrEmpty(params, "name"), GetOrEmpty(params, "value"),
                 GetOrEmpty(params, "context"));
  } else if (command == "ucinewgame") {
    CmdUciNewGame();
  } else if (command == "position") {
    if (ContainsKey(params, "fen") == ContainsKey(params, "startpos")) {
      throw Exception("Position requires either fen or startpos");
    }
    const std::vector<std::string> moves =
        StrSplitAtWhitespace(GetOrEmpty(params, "moves"));
    CmdPosition(GetOrEmpty(params, "fen"), moves);
  } else if (command == "go") {
    GoParams go_params;
    if (ContainsKey(params, "infinite")) {
      if (!GetOrEmpty(params, "infinite").empty()) {
        throw Exception("Unexpected token " + GetOrEmpty(params, "infinite"));
      }
      go_params.infinite = true;
    }
    if (ContainsKey(params, "searchmoves")) {
      go_params.searchmoves =
          StrSplitAtWhitespace(GetOrEmpty(params, "searchmoves"));
    }
    if (ContainsKey(params, "ponder")) {
      if (!GetOrEmpty(params, "ponder").empty()) {
        throw Exception("Unexpected token " + GetOrEmpty(params, "ponder"));
      }
      go_params.ponder = true;
    }
#define UCIGOOPTION(x)                    \
  if (ContainsKey(params, #x)) {          \
    go_params.x = GetNumeric(params, #x); \
  }
    UCIGOOPTION(wtime);
    UCIGOOPTION(btime);
    UCIGOOPTION(winc);
    UCIGOOPTION(binc);
    UCIGOOPTION(movestogo);
    UCIGOOPTION(depth);
    UCIGOOPTION(nodes);
    UCIGOOPTION(movetime);
#undef UCIGOOPTION
    CmdGo(go_params);
  } else if (command == "stop") {
    CmdStop();
  } else if (command == "ponderhit") {
    CmdPonderHit();
  } else if (command == "start") {
    CmdStart();
  } else if (command == "fen") {
    CmdFen();
  } else if (command == "xyzzy") {
    SendResponse("Nothing happens.");
  } else if (command == "quit") {
    return false;
  } else {
    throw Exception("Unknown command: " + command);
  }
  return true;
}
void UciLoop::SendResponse(const std::string& response) {
  SendResponses({response});
}
void UciLoop::SendResponses(const std::vector<std::string>& responses) {
  static std::mutex output_mutex;
  std::lock_guard<std::mutex> lock(output_mutex);
  for (auto& response : responses) {
    LOGFILE << "<< " << response;
    std::cout << response << std::endl;
  }
}
void UciLoop::SendId() {
  SendResponse("id name Lc0 v" + GetVersionStr());
  SendResponse("id author The LCZero Authors.");
}
void UciLoop::SendBestMove(const BestMoveInfo& move) {
  std::string res = "bestmove " + move.bestmove.as_string();
  if (move.ponder) res += " ponder " + move.ponder.as_string();
  if (move.player != -1) res += " player " + std::to_string(move.player);
  if (move.game_id != -1) res += " gameid " + std::to_string(move.game_id);
  if (move.is_black)
    res += " side " + std::string(*move.is_black ? "black" : "white");
  SendResponse(res);
}
void UciLoop::SendInfo(const std::vector<ThinkingInfo>& infos) {
  std::vector<std::string> reses;
  for (const auto& info : infos) {
    std::string res = "info";
    if (info.player != -1) res += " player " + std::to_string(info.player);
    if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
    if (info.is_black)
      res += " side " + std::string(*info.is_black ? "black" : "white");
    if (info.depth >= 0)
      res += " depth " + std::to_string(std::max(info.depth, 1));
    if (info.seldepth >= 0) res += " seldepth " + std::to_string(info.seldepth);
    if (info.time >= 0) res += " time " + std::to_string(info.time);
    if (info.nodes >= 0) res += " nodes " + std::to_string(info.nodes);
    if (info.mate) res += " score mate " + std::to_string(*info.mate);
    if (info.score) res += " score cp " + std::to_string(*info.score);
    if (info.wdl) {
      res += " wdl " + std::to_string(info.wdl->w) + " " +
             std::to_string(info.wdl->d) + " " + std::to_string(info.wdl->l);
    }
    if (info.moves_left) {
      res += " movesleft " + std::to_string(*info.moves_left);
    }
    if (info.hashfull >= 0) res += " hashfull " + std::to_string(info.hashfull);
    if (info.nps >= 0) res += " nps " + std::to_string(info.nps);
    if (info.tb_hits >= 0) res += " tbhits " + std::to_string(info.tb_hits);
    if (info.multipv >= 0) res += " multipv " + std::to_string(info.multipv);
    if (!info.pv.empty()) {
      res += " pv";
      for (const auto& move : info.pv) res += " " + move.as_string();
    }
    if (!info.comment.empty()) res += " string " + info.comment;
    reses.push_back(std::move(res));
  }
  SendResponses(reses);
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/chess/uciloop.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/protomessage.cc
namespace lczero {
namespace {
uint64_t ReadVarInt(const std::uint8_t** iter, const std::uint8_t* const end) {
  uint64_t res = 0;
  uint64_t multiplier = 1;
  while (*iter < end) {
    std::uint8_t x = **iter;
    ++*iter;
    res += (x & 0x7f) * multiplier;
    if ((x & 0x80) == 0) return res;
    multiplier *= 0x80;
  }
  throw Exception("The file seems truncated.");
}
void CheckOutOfBounds(const std::uint8_t* const iter, size_t size,
                      const std::uint8_t* const end) {
  if (iter + size > end) {
    throw Exception("The file is truncated.");
  }
}
uint64_t ReadFixed(const std::uint8_t** iter, size_t size,
                   const std::uint8_t* const end) {
  CheckOutOfBounds(*iter, size, end);
  uint64_t multiplier = 1;
  uint64_t result = 0;
  for (; size != 0; --size, multiplier *= 256, ++*iter) {
    result += multiplier * **iter;
  }
  return result;
}
void WriteFixed(uint64_t value, size_t size, std::string* out) {
  out->reserve(out->size() + size);
  for (size_t i = 0; i < size; ++i) {
    out->push_back(static_cast<char>(static_cast<uint8_t>(value)));
    value /= 256;
  }
}
// // Kept for serialization part.
std::string EncodeVarInt(std::uint64_t val) {
  std::string res;
  while (true) {
    char c = (val & 0x7f);
    val >>= 7;
    if (val) c |= 0x80;
    res += c;
    if (!val) return res;
  }
}
}  // namespace
void ProtoMessage::ParseFromString(std::string_view str) {
  Clear();
  return MergeFromString(str);
}
void ProtoMessage::MergeFromString(std::string_view str) {
  const std::uint8_t* iter = reinterpret_cast<const std::uint8_t*>(str.data());
  const std::uint8_t* const end = iter + str.size();
  while (iter < end) {
    uint64_t wire_field_id = ReadVarInt(&iter, end);
    uint64_t field_id = wire_field_id >> 3;
    switch (wire_field_id & 0x7) {
      case 0:
        // Varint field, so read one more varint.
        SetVarInt(field_id, ReadVarInt(&iter, end));
        break;
      case 1:
        // Fixed64, read 8 bytes.
        SetInt64(field_id, ReadFixed(&iter, 8, end));
        break;
      case 2: {
        // String/submessage. Varint length and then buffer of that length.
        size_t size = ReadVarInt(&iter, end);
        CheckOutOfBounds(iter, size, end);
        SetString(field_id,
                  std::string_view(reinterpret_cast<const char*>(iter), size));
        iter += size;
        break;
      }
      case 5:
        // Fixed32, read 4 bytes.
        SetInt32(field_id, ReadFixed(&iter, 4, end));
        break;
      default:
        throw Exception("The file seems to be unparseable.");
    }
  }
}
void ProtoMessage::AppendVarInt(int field_id, std::uint64_t value,
                                std::string* out) const {
  *out += EncodeVarInt(field_id << 3);
  *out += EncodeVarInt(value);
}
void ProtoMessage::AppendInt64(int field_id, std::uint64_t value,
                               std::string* out) const {
  *out += EncodeVarInt(1 + (field_id << 3));
  WriteFixed(value, 8, out);
}
void ProtoMessage::AppendInt32(int field_id, std::uint32_t value,
                               std::string* out) const {
  *out += EncodeVarInt(5 + (field_id << 3));
  WriteFixed(value, 4, out);
}
void ProtoMessage::AppendString(int field_id, std::string_view value,
                                std::string* out) const {
  *out += EncodeVarInt(2 + (field_id << 3));
  *out += EncodeVarInt(value.size());
  *out += value;
}
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/utils/protomessage.cc

// begin of /Users/syys/CLionProjects/lc0/src/neural/cache.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
CachingComputation::CachingComputation(
    std::unique_ptr<NetworkComputation> parent, NNCache* cache)
    : parent_(std::move(parent)), cache_(cache) {}
int CachingComputation::GetCacheMisses() const {
  return parent_->GetBatchSize();
}
int CachingComputation::GetBatchSize() const { return batch_.size(); }
bool CachingComputation::AddInputByHash(uint64_t hash) {
  NNCacheLock lock(cache_, hash);
  if (!lock) return false;
  AddInputByHash(hash, std::move(lock));
  return true;
}
void CachingComputation::AddInputByHash(uint64_t hash, NNCacheLock&& lock) {
  assert(lock);
  batch_.emplace_back();
  batch_.back().lock = std::move(lock);
  batch_.back().hash = hash;
}
void CachingComputation::PopCacheHit() {
  assert(!batch_.empty());
  assert(batch_.back().lock);
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}
void CachingComputation::AddInput(
    uint64_t hash, InputPlanes&& input,
    std::vector<uint16_t>&& probabilities_to_cache) {
  if (AddInputByHash(hash)) return;
  batch_.emplace_back();
  batch_.back().hash = hash;
  batch_.back().idx_in_parent = parent_->GetBatchSize();
  batch_.back().probabilities_to_cache = probabilities_to_cache;
  parent_->AddInput(std::move(input));
}
void CachingComputation::PopLastInputHit() {
  assert(!batch_.empty());
  assert(batch_.back().idx_in_parent == -1);
  batch_.pop_back();
}
void CachingComputation::ComputeBlocking() {
  if (parent_->GetBatchSize() == 0) return;
  parent_->ComputeBlocking();
  // Fill cache with data from NN.
  for (const auto& item : batch_) {
    if (item.idx_in_parent == -1) continue;
    auto req =
        std::make_unique<CachedNNRequest>(item.probabilities_to_cache.size());
    req->q = parent_->GetQVal(item.idx_in_parent);
    req->d = parent_->GetDVal(item.idx_in_parent);
    req->m = parent_->GetMVal(item.idx_in_parent);
    int idx = 0;
    for (auto x : item.probabilities_to_cache) {
      req->p[idx++] =
          std::make_pair(x, parent_->GetPVal(item.idx_in_parent, x));
    }
    cache_->Insert(item.hash, std::move(req));
  }
}
float CachingComputation::GetQVal(int sample) const {
  const auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) return parent_->GetQVal(item.idx_in_parent);
  return item.lock->q;
}
float CachingComputation::GetDVal(int sample) const {
  const auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) return parent_->GetDVal(item.idx_in_parent);
  return item.lock->d;
}
float CachingComputation::GetMVal(int sample) const {
  const auto& item = batch_[sample];
  if (item.idx_in_parent >= 0) return parent_->GetMVal(item.idx_in_parent);
  return item.lock->m;
}
float CachingComputation::GetPVal(int sample, int move_id) const {
  auto& item = batch_[sample];
  if (item.idx_in_parent >= 0)
    return parent_->GetPVal(item.idx_in_parent, move_id);
  const auto& moves = item.lock->p;
  int total_count = 0;
  while (total_count < moves.size()) {
    // Optimization: usually moves are stored in the same order as queried.
    const auto& move = moves[item.last_idx++];
    if (item.last_idx == moves.size()) item.last_idx = 0;
    if (move.first == move_id) return move.second;
    ++total_count;
  }
  assert(false);  // Move not found.
  return 0;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/cache.cc

// begin of /Users/syys/CLionProjects/lc0/src/neural/encoder.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
int CompareTransposing(BitBoard board, int initial_transform) {
  uint64_t value = board.as_int();
  if ((initial_transform & FlipTransform) != 0) {
    value = ReverseBitsInBytes(value);
  }
  if ((initial_transform & MirrorTransform) != 0) {
    value = ReverseBytesInBytes(value);
  }
  auto alternative = TransposeBitsInBytes(value);
  if (value < alternative) return -1;
  if (value > alternative) return 1;
  return 0;
}
int ChooseTransform(const ChessBoard& board) {
  // If there are any castling options no transform is valid.
  // Even using FRC rules, king and queen side castle moves are not symmetrical.
  if (!board.castlings().no_legal_castle()) {
    return 0;
  }
  auto our_king = (board.kings() & board.ours()).as_int();
  int transform = NoTransform;
  if ((our_king & 0x0F0F0F0F0F0F0F0FULL) != 0) {
    transform |= FlipTransform;
    our_king = ReverseBitsInBytes(our_king);
  }
  // If there are any pawns only horizontal flip is valid.
  if (board.pawns().as_int() != 0) {
    return transform;
  }
  if ((our_king & 0xFFFFFFFF00000000ULL) != 0) {
    transform |= MirrorTransform;
    our_king = ReverseBytesInBytes(our_king);
  }
  // Our king is now always in bottom right quadrant.
  // Transpose for king in top right triangle, or if on diagonal whichever has
  // the smaller integer value for each test scenario.
  if ((our_king & 0xE0C08000ULL) != 0) {
    transform |= TransposeTransform;
  } else if ((our_king & 0x10204080ULL) != 0) {
    auto outcome = CompareTransposing(board.ours() | board.theirs(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.ours(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.kings(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.queens(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.rooks(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.knights(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.bishops(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    // If all piece types are symmetrical and ours is symmetrical and
    // ours+theirs is symmetrical, everything is symmetrical, so transpose is a
    // no-op.
  }
  return transform;
}
}  // namespace
bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION;
}
bool IsCanonicalArmageddonFormat(
    pblczero::NetworkFormat::InputFormat input_format) {
  return input_format ==
             pblczero::NetworkFormat::
                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == pblczero::NetworkFormat::
                             INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES;
}
bool Is960CastlingFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >= pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE;
}
int TransformForPosition(pblczero::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history) {
  if (!IsCanonicalFormat(input_format)) {
    return 0;
  }
  const ChessBoard& board = history.Last().GetBoard();
  return ChooseTransform(board);
}
InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out) {
  InputPlanes result(kAuxPlaneBase + 8);
  int transform = 0;
  // Canonicalization format needs to stop early to avoid applying transform in
  // history across incompatible transitions.  It is also more canonical since
  // history before these points is not relevant to the final result.
  bool stop_early = IsCanonicalFormat(input_format);
  // When stopping early, we want to know if castlings has changed, so capture
  // it for the first board.
  ChessBoard::Castlings castlings;
  {
    const ChessBoard& board = history.Last().GetBoard();
    const bool we_are_black = board.flipped();
    if (IsCanonicalFormat(input_format)) {
      transform = ChooseTransform(board);
    }
    switch (input_format) {
      case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE: {
        // "Legacy" input planes with:
        // - Plane 104 (0-based) filled with 1 if white can castle queenside.
        // - Plane 105 filled with ones if white can castle kingside.
        // - Plane 106 filled with ones if black can castle queenside.
        // - Plane 107 filled with ones if white can castle kingside.
        if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
        if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
        if (board.castlings().they_can_000()) {
          result[kAuxPlaneBase + 2].SetAll();
        }
        if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
        break;
      }
      case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
      case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
      case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
      case pblczero::NetworkFormat::
          INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
      case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
      case pblczero::NetworkFormat::
          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
        // - Plane 104 for positions of rooks (both white and black) which
        // have
        // a-side (queenside) castling right.
        // - Plane 105 for positions of rooks (both white and black) which have
        // h-side (kingside) castling right.
        const auto& cast = board.castlings();
        result[kAuxPlaneBase + 0].mask =
            ((cast.we_can_000() ? BoardSquare(ChessBoard::A1).as_board() : 0) |
             (cast.they_can_000() ? BoardSquare(ChessBoard::A8).as_board() : 0))
            << cast.queenside_rook();
        result[kAuxPlaneBase + 1].mask =
            ((cast.we_can_00() ? BoardSquare(ChessBoard::A1).as_board() : 0) |
             (cast.they_can_00() ? BoardSquare(ChessBoard::A8).as_board() : 0))
            << cast.kingside_rook();
        break;
      }
      default:
        throw Exception("Unsupported input plane encoding " +
                        std::to_string(input_format));
    };
    if (IsCanonicalFormat(input_format)) {
      result[kAuxPlaneBase + 4].mask = board.en_passant().as_int();
    } else {
      if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
    }
    if (IsHectopliesFormat(input_format)) {
      result[kAuxPlaneBase + 5].Fill(history.Last().GetRule50Ply() / 100.0f);
    } else {
      result[kAuxPlaneBase + 5].Fill(history.Last().GetRule50Ply());
    }
    // Plane kAuxPlaneBase + 6 used to be movecount plane, now it's all zeros
    // unless we need it for canonical armageddon side to move.
    if (IsCanonicalArmageddonFormat(input_format)) {
      if (we_are_black) result[kAuxPlaneBase + 6].SetAll();
    }
    // Plane kAuxPlaneBase + 7 is all ones to help NN find board edges.
    result[kAuxPlaneBase + 7].SetAll();
    if (stop_early) {
      castlings = board.castlings();
    }
  }
  bool skip_non_repeats =
      input_format ==
          pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
      input_format == pblczero::NetworkFormat::
                          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
  bool flip = false;
  int history_idx = history.GetLength() - 1;
  for (int i = 0; i < std::min(history_planes, kMoveHistory);
       ++i, --history_idx) {
    const Position& position =
        history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    // Castling changes can't be repeated, so we can stop early.
    if (stop_early && board.castlings().as_int() != castlings.as_int()) break;
    // Enpassants can't be repeated, but we do need to always send the current
    // position.
    if (stop_early && history_idx != history.GetLength() - 1 &&
        !board.en_passant().empty()) {
      break;
    }
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
    // Board may be flipped so compare with position.GetBoard().
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        position.GetBoard() == ChessBoard::kStartposBoard) {
      break;
    }
    const int repetitions = position.GetRepetitions();
    // Canonical v2 only writes an item if it is a repeat, unless its the most
    // recent position.
    if (skip_non_repeats && repetitions == 0 && i > 0) {
      if (history_idx > 0) flip = !flip;
      // If no capture no pawn is 0, the previous was start of game, capture or
      // pawn push, so there can't be any more repeats that are worth
      // considering.
      if (position.GetRule50Ply() == 0) break;
      // Decrement i so it remains the same as the history_idx decrements.
      --i;
      continue;
    }
    const int base = i * kPlanesPerBoard;
    result[base + 0].mask = (board.ours() & board.pawns()).as_int();
    result[base + 1].mask = (board.ours() & board.knights()).as_int();
    result[base + 2].mask = (board.ours() & board.bishops()).as_int();
    result[base + 3].mask = (board.ours() & board.rooks()).as_int();
    result[base + 4].mask = (board.ours() & board.queens()).as_int();
    result[base + 5].mask = (board.ours() & board.kings()).as_int();
    result[base + 6].mask = (board.theirs() & board.pawns()).as_int();
    result[base + 7].mask = (board.theirs() & board.knights()).as_int();
    result[base + 8].mask = (board.theirs() & board.bishops()).as_int();
    result[base + 9].mask = (board.theirs() & board.rooks()).as_int();
    result[base + 10].mask = (board.theirs() & board.queens()).as_int();
    result[base + 11].mask = (board.theirs() & board.kings()).as_int();
    if (repetitions >= 1) result[base + 12].SetAll();
    // If en passant flag is set, undo last pawn move by removing the pawn from
    // the new square and putting into pre-move square.
    if (history_idx < 0 && !board.en_passant().empty()) {
      const auto idx = GetLowestBit(board.en_passant().as_int());
      if (idx < 8) {  // "Us" board
        result[base + 0].mask +=
            ((0x0000000000000100ULL - 0x0000000001000000ULL) << idx);
      } else {
        result[base + 6].mask +=
            ((0x0001000000000000ULL - 0x0000000100000000ULL) << (idx - 56));
      }
    }
    if (history_idx > 0) flip = !flip;
    // If no capture no pawn is 0, the previous was start of game, capture or
    // pawn push, so no need to go back further if stopping early.
    if (stop_early && position.GetRule50Ply() == 0) break;
  }
  if (transform != NoTransform) {
    // Transform all masks.
    for (int i = 0; i <= kAuxPlaneBase + 4; i++) {
      auto v = result[i].mask;
      if (v == 0 || v == ~0ULL) continue;
      if ((transform & FlipTransform) != 0) {
        v = ReverseBitsInBytes(v);
      }
      if ((transform & MirrorTransform) != 0) {
        v = ReverseBytesInBytes(v);
      }
      if ((transform & TransposeTransform) != 0) {
        v = TransposeBitsInBytes(v);
      }
      result[i].mask = v;
    }
  }
  if (transform_out) *transform_out = transform;
  return result;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/encoder.cc

// begin of /Users/syys/CLionProjects/lc0/src/mcts/node.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
/////////////////////////////////////////////////////////////////////////
// Node garbage collector
/////////////////////////////////////////////////////////////////////////
namespace {
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;
// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}
  // Takes ownership of a subtree, to dispose it in a separate thread when
  // it has time.
  void AddToGcQueue(std::unique_ptr<Node> node, size_t solid_size = 0) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
    subtrees_to_gc_solid_size_.push_back(solid_size);
  }
  ~NodeGarbageCollector() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_.store(true);
    gc_thread_.join();
  }
 private:
  void GarbageCollect() {
    while (!stop_.load()) {
      // Node will be released in destructor when mutex is not locked.
      std::unique_ptr<Node> node_to_gc;
      size_t solid_size = 0;
      {
        // Lock the mutex and move last subtree from subtrees_to_gc_ into
        // node_to_gc.
        Mutex::Lock lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
        solid_size = subtrees_to_gc_solid_size_.back();
        subtrees_to_gc_solid_size_.pop_back();
      }
      // Solid is a hack...
      if (solid_size != 0) {
        for (size_t i = 0; i < solid_size; i++) {
          node_to_gc.get()[i].~Node();
        }
        std::allocator<Node> alloc;
        alloc.deallocate(node_to_gc.release(), solid_size);
      }
    }
  }
  void Worker() {
    // Keep garbage collection on same core as where search workers are most
    // likely to be to make any lock conention on gc mutex cheaper.
    Numa::BindThread(0);
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
  }
  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_ GUARDED_BY(gc_mutex_);
  std::vector<size_t> subtrees_to_gc_solid_size_ GUARDED_BY(gc_mutex_);
  // When true, Worker() should stop and exit.
  std::atomic<bool> stop_{false};
  std::thread gc_thread_;
};
NodeGarbageCollector gNodeGc;
}  // namespace
/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////
Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Mirror();
  return m;
}
// Policy priors (P) are stored in a compressed 16-bit format.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)). This is fine because the values in [0,1] are skewed towards
// 0, which is also exactly how the components of policy tend to behave (since
// they add up to 1).
// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.
void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}
float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}
std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.as_string() << " p_: " << p_ << " GetP: " << GetP();
  return oss.str();
}
std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}
/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////
Node* Node::CreateSingleChildNode(Move move) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist({move});
  num_edges_ = 1;
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}
void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist(moves);
  num_edges_ = moves.size();
}
Node::ConstIterator Node::Edges() const {
  return {*this, !solid_children_ ? &child_ : nullptr};
}
Node::Iterator Node::Edges() {
  return {*this, !solid_children_ ? &child_ : nullptr};
}
float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += GetEdgeToNode(node)->GetP();
  return sum;
}
Edge* Node::GetEdgeToNode(const Node* node) const {
  assert(node->parent_ == this);
  assert(node->index_ < num_edges_);
  return &edges_[node->index_];
}
Edge* Node::GetOwnEdge() const { return GetParent()->GetEdgeToNode(this); }
std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << static_cast<int>(terminal_type_) << " This:" << this
      << " Parent:" << parent_ << " Index:" << index_
      << " Child:" << child_.get() << " Sibling:" << sibling_.get()
      << " WL:" << wl_ << " N:" << n_ << " N_:" << n_in_flight_
      << " Edges:" << static_cast<int>(num_edges_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << " Solid:" << solid_children_;
  return oss.str();
}
bool Node::MakeSolid() {
  if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;
  // Can only make solid if no immediate leaf childredn are in flight since we
  // allow the search code to hold references to leaf nodes across locks.
  Node* old_child_to_check = child_.get();
  uint32_t total_in_flight = 0;
  while (old_child_to_check != nullptr) {
    if (old_child_to_check->GetN() <= 1 &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    if (old_child_to_check->IsTerminal() &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    total_in_flight += old_child_to_check->GetNInFlight();
    old_child_to_check = old_child_to_check->sibling_.get();
  }
  // If the total of children in flight is not the same as self, then there are
  // collisions against immediate children (which don't update the GetNInFlight
  // of the leaf) and its not safe.
  if (total_in_flight != GetNInFlight()) {
    return false;
  }
  std::allocator<Node> alloc;
  auto* new_children = alloc.allocate(num_edges_);
  for (int i = 0; i < num_edges_; i++) {
    new (&(new_children[i])) Node(this, i);
  }
  std::unique_ptr<Node> old_child = std::move(child_);
  while (old_child) {
    int index = old_child->index_;
    new_children[index] = std::move(*old_child.get());
    // This isn't needed, but it helps crash things faster if something has gone wrong.
    old_child->parent_ = nullptr;
    gNodeGc.AddToGcQueue(std::move(old_child));
    new_children[index].UpdateChildrenParents();
    old_child = std::move(new_children[index].sibling_);
  }
  // This is a hack.
  child_ = std::unique_ptr<Node>(new_children);
  solid_children_ = true;
  return true;
}
void Node::SortEdges() {
  assert(edges_);
  assert(!child_);
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges_.get(), (edges_.get() + num_edges_),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}
void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  if (type != Terminal::TwoFold) SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    // Terminal losses have no uncertainty and no reason for their U value to be
    // comparable to another non-loss choice. Force this by clearing the policy.
    if (GetParent() != nullptr) GetOwnEdge()->SetP(0.0f);
  }
}
void Node::MakeNotTerminal() {
  terminal_type_ = Terminal::NonTerminal;
  n_ = 0;
  // If we have edges, we've been extended (1 visit), so include children too.
  if (edges_) {
    n_++;
    for (const auto& child : Edges()) {
      const auto n = child.GetN();
      if (n > 0) {
        n_ += n;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += -child.GetWL(0.0f) * n;
        d_ += child.GetD(0.0f) * n;
      }
    }
    // Recompute with current eval (instead of network's) and children's eval.
    wl_ /= n_;
    d_ /= n_;
  }
}
void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}
bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}
void Node::CancelScoreUpdate(int multivisit) {
  n_in_flight_ -= multivisit;
}
void Node::FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
  // Recompute Q.
  wl_ += multivisit * (v - wl_) / (n_ + multivisit);
  d_ += multivisit * (d - d_) / (n_ + multivisit);
  m_ += multivisit * (m - m_) / (n_ + multivisit);
  // Increment N.
  n_ += multivisit;
  // Decrement virtual loss.
  n_in_flight_ -= multivisit;
}
void Node::AdjustForTerminal(float v, float d, float m, int multivisit) {
  // Recompute Q.
  wl_ += multivisit * v / n_;
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;
}
void Node::RevertTerminalVisits(float v, float d, float m, int multivisit) {
  // Compute new n_ first, as reducing a node to 0 visits is a special case.
  const int n_new = n_ - multivisit;
  if (n_new <= 0) {
    // If n_new == 0, reset all relevant values to 0.
    wl_ = 0.0;
    d_ = 1.0;
    m_ = 0.0;
    n_ = 0;
  } else {
    // Recompute Q and M.
    wl_ -= multivisit * (v - wl_) / n_new;
    d_ -= multivisit * (d - d_) / n_new;
    m_ -= multivisit * (m - m_) / n_new;
    // Decrement N.
    n_ -= multivisit;
  }
}
void Node::UpdateChildrenParents() {
  if (!solid_children_) {
    Node* cur_child = child_.get();
    while (cur_child != nullptr) {
      cur_child->parent_ = this;
      cur_child = cur_child->sibling_.get();
    }
  } else {
    Node* child_array = child_.get();
    for (int i = 0; i < num_edges_; i++) {
      child_array[i].parent_ = this;
    }
  }
}
void Node::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_), solid_children_ ? num_edges_ : 0);
}
void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  if (solid_children_) {
    std::unique_ptr<Node> saved_node;
    if (node_to_save != nullptr) {
      saved_node = std::make_unique<Node>(this, node_to_save->index_);
      *saved_node = std::move(*node_to_save);
    }
    gNodeGc.AddToGcQueue(std::move(child_), num_edges_);
    child_ = std::move(saved_node);
    if (child_) {
      child_->UpdateChildrenParents();
    }
    solid_children_ = false;
  } else {
    // Stores node which will have to survive (or nullptr if it's not found).
    std::unique_ptr<Node> saved_node;
    // Pointer to unique_ptr, so that we could move from it.
    for (std::unique_ptr<Node>* node = &child_; *node;
         node = &(*node)->sibling_) {
      // If current node is the one that we have to save.
      if (node->get() == node_to_save) {
        // Kill all remaining siblings.
        gNodeGc.AddToGcQueue(std::move((*node)->sibling_));
        // Save the node, and take the ownership from the unique_ptr.
        saved_node = std::move(*node);
        break;
      }
    }
    // Make saved node the only child. (kills previous siblings).
    gNodeGc.AddToGcQueue(std::move(child_));
    child_ = std::move(saved_node);
  }
  if (!child_) {
    num_edges_ = 0;
    edges_.reset();  // Clear edges list.
  }
}
/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////
std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}
/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////
void NodeTree::MakeMove(Move move) {
  if (HeadPosition().IsBlackToMove()) move.Mirror();
  const auto& board = HeadPosition().GetBoard();
  Node* new_head = nullptr;
  for (auto& n : current_head_->Edges()) {
    if (board.IsSameMove(n.GetMove(), move)) {
      new_head = n.GetOrSpawnNode(current_head_);
      // Ensure head is not terminal, so search can extend or visit children of
      // "terminal" positions, e.g., WDL hits, converted terminals, 3-fold draw.
      if (new_head->IsTerminal()) new_head->MakeNotTerminal();
      break;
    }
  }
  move = board.GetModernMove(move);
  current_head_->ReleaseChildrenExceptOne(new_head);
  new_head = current_head_->child_.get();
  current_head_ =
      new_head ? new_head : current_head_->CreateSingleChildNode(move);
  history_.Append(move);
}
void NodeTree::TrimTreeAtHead() {
  // If solid, this will be empty before move and will be moved back empty
  // afterwards which is fine.
  auto tmp = std::move(current_head_->sibling_);
  // Send dependent nodes for GC instead of destroying them immediately.
  current_head_->ReleaseChildren();
  *current_head_ = Node(current_head_->GetParent(), current_head_->index_);
  current_head_->sibling_ = std::move(tmp);
}
bool NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (gamebegin_node_ &&
      (history_.Starting().GetBoard() != starting_board ||
       history_.Starting().GetRule50Ply() != no_capture_ply)) {
    // Completely different position.
    DeallocateTree();
  }
  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>(nullptr, 0);
  }
  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));
  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }
  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  if (!seen_old_head) TrimTreeAtHead();
  return seen_old_head;
}
void NodeTree::DeallocateTree() {
  // Same as gamebegin_node_.reset(), but actual deallocation will happen in
  // GC thread.
  gNodeGc.AddToGcQueue(std::move(gamebegin_node_));
  gamebegin_node_ = nullptr;
  current_head_ = nullptr;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/node.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/optionsdict.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
const OptionsDict& OptionsDict::GetSubdict(const std::string& name) const {
  const auto iter = subdicts_.find(name);
  if (iter == subdicts_.end())
    throw Exception("Subdictionary not found: " + name);
  return iter->second;
}
// Returns subdictionary. Throws exception if doesn't exist.
OptionsDict* OptionsDict::GetMutableSubdict(const std::string& name) {
  auto iter = subdicts_.find(name);
  if (iter == subdicts_.end())
    throw Exception("Subdictionary not found: " + name);
  return &iter->second;
}
// Creates subdictionary. Throws exception if already exists.
OptionsDict* OptionsDict::AddSubdict(const std::string& name) {
  const auto iter = subdicts_.find(name);
  if (iter != subdicts_.end())
    throw Exception("Subdictionary already exists: " + name);
  const auto x = &subdicts_.emplace(name, this).first->second;
  return x;
}
void OptionsDict::AddAliasDict(const OptionsDict* dict) {
  aliases_.push_back(dict);
}
// Returns list of subdictionaries.
std::vector<std::string> OptionsDict::ListSubdicts() const {
  std::vector<std::string> result;
  for (const auto& subdict : subdicts_) {
    result.emplace_back(subdict.first);
  }
  return result;
}
bool OptionsDict::HasSubdict(const std::string& name) const {
  return subdicts_.find(name) != subdicts_.end();
}
namespace {
class Lexer {
 public:
  enum TokenType {
    L_INTEGER,
    L_FLOAT,
    L_STRING,
    L_IDENTIFIER,
    L_LEFT_PARENTHESIS,
    L_RIGHT_PARENTHESIS,
    L_COMMA,
    L_EQUAL,
    L_EOF
  };
  Lexer(const std::string& str) : str_(str) { Next(); }
  void Next() {
    // Skip whitespace:
    while (idx_ < str_.size() && std::isspace(str_[idx_])) ++idx_;
    last_offset_ = idx_;
    // If end of line, report end of line.
    if (idx_ == str_.size()) {
      type_ = L_EOF;
      return;
    }
    // Single characters.
    static const std::pair<char, TokenType> kCharacters[] = {
        {',', L_COMMA},
        {'(', L_LEFT_PARENTHESIS},
        {')', L_RIGHT_PARENTHESIS},
        {'=', L_EQUAL}};
    for (const auto& ch : kCharacters) {
      if (str_[idx_] == ch.first) {
        ++idx_;
        type_ = ch.second;
        return;
      }
    }
    // Numbers (integer of float).
    static const std::string kNumberChars = "0123456789-.";
    if (kNumberChars.find(str_[idx_]) != std::string::npos) {
      ReadNumber();
      return;
    }
    // Strings (single or double quoted)
    if (str_[idx_] == '\'' || str_[idx_] == '\"') {
      ReadString();
      return;
    }
    // Identifier
    if (std::isalnum(str_[idx_]) || str_[idx_] == '/') {
      ReadIdentifier();
      return;
    }
    RaiseError("Unable to parse token");
  }
  void RaiseError(const std::string& message) {
    throw Exception("Unable to parse config at offset " +
                    std::to_string(last_offset_) + ": " + str_ + " (" +
                    message + ")");
  }
  TokenType GetToken() const { return type_; }
  const std::string& GetStringVal() const { return string_val_; }
  int GetIntVal() const { return int_val_; }
  float GetFloatVal() const { return float_val_; }
 private:
  void ReadString() {
    last_offset_ = idx_;
    const char quote = str_[idx_++];
    for (; idx_ < str_.size(); ++idx_) {
      if (str_[idx_] == quote) {
        type_ = L_STRING;
        string_val_ = str_.substr(last_offset_ + 1, idx_ - last_offset_ - 1);
        ++idx_;
        return;
      }
    }
    last_offset_ = idx_;
    RaiseError("String is not closed at end of line");
  }
  void ReadIdentifier() {
    string_val_ = "";
    type_ = L_IDENTIFIER;
    static const std::string kAllowedPunctuation = "_-./";
    for (; idx_ < str_.size(); ++idx_) {
      if (!std::isalnum(str_[idx_]) &&
          kAllowedPunctuation.find(str_[idx_]) == std::string::npos) {
        break;
      }
      string_val_ += str_[idx_];
    }
  }
  void ReadNumber() {
    last_offset_ = idx_;
    bool is_float = false;
    static const std::string kFloatChars = ".eE";
    static const std::string kAllowedChars = "+-1234567890.eExX";
    for (; idx_ < str_.size(); ++idx_) {
      if (kAllowedChars.find(str_[idx_]) == std::string::npos) break;
      if (kFloatChars.find(str_[idx_]) != std::string::npos) is_float = true;
    }
    try {
      if (is_float) {
        type_ = L_FLOAT;
        float_val_ = stof(str_.substr(last_offset_, idx_ - last_offset_));
      } else {
        type_ = L_INTEGER;
        int_val_ = stoi(str_.substr(last_offset_, idx_ - last_offset_));
      }
    } catch (...) {
      RaiseError("Unable to parse number");
    }
  }
  float float_val_;
  int int_val_;
  std::string string_val_;
  TokenType type_;
  const std::string str_;
  size_t idx_ = 0;
  int last_offset_ = 0;
};
class Parser {
 public:
  Parser(const std::string& str) : lexer_(str) {}
  void ParseMain(OptionsDict* dict) {
    ParseList(dict);            // Parse list of options
    EnsureToken(Lexer::L_EOF);  // Check that everything is read.
  }
 private:
  // Returns first non-existing subdict with name like "[0]", "[24]", etc.
  static std::string GetFreeSubdictName(OptionsDict* dict) {
    for (int idx = 0;; ++idx) {
      std::string id = "[" + std::to_string(idx) + "]";
      if (!dict->HasSubdict(id)) return id;
    }
    assert(false);
    return "";
  }
  // Parses comma separated list of either:
  // * key=value, or
  // * subdict(comma separated list)
  // Note that in subdict all parts are optional:
  // * (comma separated list) -- name will be synthesized (e.g. "[1]")
  // * subdict() -- empty list
  // * subdict -- the same.
  void ParseList(OptionsDict* dict) {
    while (true) {
      std::string identifier;
      if (lexer_.GetToken() == Lexer::L_LEFT_PARENTHESIS) {
        // List entry starts with "(", that's a special case of subdict without
        // name, we have to come up with the name ourselves.
        identifier = GetFreeSubdictName(dict);
      } else if (lexer_.GetToken() == Lexer::L_IDENTIFIER ||
                 lexer_.GetToken() == Lexer::L_STRING) {
        // Read identifier.
        identifier = lexer_.GetStringVal();
        lexer_.Next();
      } else {
        // Unexpected token, exiting parsing list.
        return;
      }
      // If there is "=" after identifier, that's key=value entry, read value.
      if (lexer_.GetToken() == Lexer::L_EQUAL) {
        lexer_.Next();
        ReadVal(dict, identifier);
      } else {
        // Otherwise it's subdict.
        ReadSubDict(dict, identifier);
      }
      // If next val is not comma, end of the list.
      if (lexer_.GetToken() != Lexer::L_COMMA) return;
      lexer_.Next();
    }
  }
  void EnsureToken(Lexer::TokenType type) {
    if (lexer_.GetToken() != type)
      lexer_.RaiseError("Expected token #" + std::to_string(type));
  }
  void ReadVal(OptionsDict* dict, const std::string& id) {
    if (lexer_.GetToken() == Lexer::L_FLOAT) {
      dict->Set<float>(id, lexer_.GetFloatVal());
    } else if (lexer_.GetToken() == Lexer::L_INTEGER) {
      dict->Set<int>(id, lexer_.GetIntVal());
    } else if (lexer_.GetToken() == Lexer::L_STRING) {
      // Strings may be:
      // * Single quoted: 'asdf'
      // * Double quoted: "asdf"
      // * Without quotes, if only alphanumeric and not "true" or "false".
      dict->Set<std::string>(id, lexer_.GetStringVal());
    } else if (lexer_.GetToken() == Lexer::L_IDENTIFIER) {
      if (lexer_.GetStringVal() == "true") {
        dict->Set<bool>(id, true);
      } else if (lexer_.GetStringVal() == "false") {
        dict->Set<bool>(id, false);
      } else {
        dict->Set<std::string>(id, lexer_.GetStringVal());
      }
    } else {
      lexer_.RaiseError("Expected value");
    }
    lexer_.Next();
  }
  void ReadSubDict(OptionsDict* dict, const std::string& identifier) {
    OptionsDict* new_dict = dict->AddSubdict(identifier);
    // If opening parentheses, read list of a subdict, otherwise list is empty,
    // so return immediately.
    if (lexer_.GetToken() == Lexer::L_LEFT_PARENTHESIS) {
      lexer_.Next();
      ParseList(new_dict);
      EnsureToken(Lexer::L_RIGHT_PARENTHESIS);
      lexer_.Next();
    }
  }
 private:
  Lexer lexer_;
};
}  // namespace
void OptionsDict::AddSubdictFromString(const std::string& str) {
  Parser parser(str);
  parser.ParseMain(this);
}
void OptionsDict::CheckAllOptionsRead(
    const std::string& path_from_parent) const {
  std::string s = path_from_parent.empty() ? "" : path_from_parent + '.';
  TypeDict<bool>::EnsureNoUnusedOptions("boolean", s);
  TypeDict<int>::EnsureNoUnusedOptions("integer", s);
  TypeDict<float>::EnsureNoUnusedOptions("floating point", s);
  TypeDict<std::string>::EnsureNoUnusedOptions("string", s);
  for (auto const& dict : subdicts_) {
    dict.second.CheckAllOptionsRead(s + dict.first);
  }
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/optionsdict.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/optionsparser.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#if __has_include(<charconv>)
#else
#define NO_CHARCONV
#endif
namespace lczero {
namespace {
const int kHelpIndent = 15;
const int kUciLineIndent = 15;
const int kHelpWidth = 80;
}  // namespace
OptionsParser::Option::Option(const OptionId& id) : id_(id) {}
OptionsParser::OptionsParser() : values_(*defaults_.AddSubdict("values")) {}
std::vector<std::string> OptionsParser::ListOptionsUci() const {
  std::vector<std::string> result;
  for (const auto& iter : options_) {
    if (!iter->GetUciOption().empty() && !iter->hidden_) {
      result.emplace_back("option name " + iter->GetUciOption() + " " +
                          iter->GetOptionString(values_));
    }
  }
  return result;
}
void OptionsParser::SetUciOption(const std::string& name,
                                 const std::string& value,
                                 const std::string& context) {
  auto option = FindOptionByUciName(name);
  if (option) {
    option->SetValue(value, GetMutableOptions(context));
    return;
  }
  throw Exception("Unknown option: " + name);
}
void OptionsParser::HideOption(const OptionId& id) {
  const auto option = FindOptionById(id);
  if (option) option->hidden_ = true;
}
OptionsParser::Option* OptionsParser::FindOptionByLongFlag(
    const std::string& flag) const {
  for (const auto& val : options_) {
    auto longflg = val->GetLongFlag();
    if (flag == longflg || flag == ("no-" + longflg)) return val.get();
  }
  return nullptr;
}
OptionsParser::Option* OptionsParser::FindOptionByUciName(
    const std::string& name) const {
  for (const auto& val : options_) {
    if (StringsEqualIgnoreCase(val->GetUciOption(), name)) return val.get();
  }
  return nullptr;
}
OptionsParser::Option* OptionsParser::FindOptionById(const OptionId& id) const {
  for (const auto& val : options_) {
    if (id == val->GetId()) return val.get();
  }
  return nullptr;
}
OptionsDict* OptionsParser::GetMutableOptions(const std::string& context) {
  if (context == "") return &values_;
  auto* result = &values_;
  for (const auto& x : StrSplit(context, ".")) {
    result = result->GetMutableSubdict(x);
  }
  return result;
}
const OptionsDict& OptionsParser::GetOptionsDict(const std::string& context) {
  if (context == "") return values_;
  const auto* result = &values_;
  for (const auto& x : StrSplit(context, ".")) {
    result = &result->GetSubdict(x);
  }
  return *result;
}
bool OptionsParser::ProcessAllFlags() {
  return ProcessFlags(ConfigFile::Arguments()) &&
         ProcessFlags(CommandLine::Arguments());
}
bool OptionsParser::ProcessFlags(const std::vector<std::string>& args) {
  auto show_help = false;
  if (CommandLine::BinaryName().find("pro") != std::string::npos) {
    ShowHidden();
  }
  for (auto iter = args.begin(), end = args.end(); iter != end; ++iter) {
    std::string param = *iter;
    if (param == "--show-hidden") {
      ShowHidden();
      continue;
    }
    if (param == "-h" || param == "--help") {
      // Set a flag so that --show-hidden after --help works.
      show_help = true;
      continue;
    }
    if (param.substr(0, 2) == "--") {
      std::string context;
      param = param.substr(2);
      std::string value;
      auto pos = param.find('=');
      if (pos != std::string::npos) {
        value = param.substr(pos + 1);
        param = param.substr(0, pos);
      }
      pos = param.rfind('.');
      if (pos != std::string::npos) {
        context = param.substr(0, pos);
        param = param.substr(pos + 1);
      }
      bool processed = false;
      Option* option = FindOptionByLongFlag(param);
      if (option &&
          option->ProcessLongFlag(param, value, GetMutableOptions(context))) {
        processed = true;
      }
      if (!processed) {
        CERR << "Unknown command line flag: " << *iter << ".";
        CERR << "For help run:\n  " << CommandLine::BinaryName() << " --help";
        return false;
      }
      continue;
    }
    if (param.size() == 2 && param[0] == '-') {
      std::string value;
      bool processed = false;
      if (iter + 1 != end) {
        value = *(iter + 1);
      }
      for (auto& option : options_) {
        if (option->ProcessShortFlag(param[1], GetMutableOptions())) {
          processed = true;
          break;
        } else if (option->ProcessShortFlagWithValue(param[1], value,
                                                     GetMutableOptions())) {
          if (!value.empty()) ++iter;
          processed = true;
          break;
        }
      }
      if (!processed) {
        CERR << "Unknown command line flag: " << *iter << ".";
        CERR << "For help run:\n  " << CommandLine::BinaryName() << " --help";
        return false;
      }
      continue;
    }
    CERR << "Unknown command line argument: " << *iter << ".\n";
    CERR << "For help run:\n  " << CommandLine::BinaryName() << " --help";
    return false;
  }
  if (show_help) {
    ShowHelp();
    return false;
  }
  return true;
}
void OptionsParser::AddContext(const std::string& context) {
  values_.AddSubdict(context);
}
namespace {
std ::string FormatFlag(char short_flag, const std::string& long_flag,
                        const std::string& help,
                        const std::string& uci_option = {},
                        const std::string& def = {}) {
  std::ostringstream oss;
  oss << "  ";
  if (short_flag) {
    oss << '-' << short_flag;
  } else {
    oss << "  ";
  }
  if (short_flag && !long_flag.empty()) {
    oss << ",  ";
  } else {
    oss << "   ";
  }
  std::string long_flag_str = "";
  if (!short_flag && long_flag.empty()) {
    long_flag_str = "(uci parameter)";
  } else {
    long_flag_str = long_flag.empty() ? "" : "--" + long_flag;
  }
  oss << long_flag_str;
  auto help_lines = FlowText(help, kHelpWidth);
  bool is_first_line = true;
  for (const auto& line : help_lines) {
    if (is_first_line) {
      is_first_line = false;
      if (long_flag_str.size() < kHelpIndent - 7) {
        oss << std::string(kHelpIndent - 7 - long_flag_str.size(), ' ') << line
            << "\n";
        continue;
      }
      oss << "\n";
    }
    oss << std::string(kHelpIndent, ' ') << line << "\n";
  }
  if (!def.empty() || !uci_option.empty()) {
    oss << std::string(kUciLineIndent, ' ') << '[';
    if (!uci_option.empty()) oss << "UCI: " << uci_option;
    if (!uci_option.empty() && !def.empty()) oss << "  ";
    if (!def.empty()) oss << "DEFAULT: " << def;
    oss << "]\n";
  }
  oss << '\n';
  return oss.str();
}
}  // namespace
void OptionsParser::ShowHelp() const {
  std::cout << "Usage: " << CommandLine::BinaryName() << " [<mode>] [flags...]"
            << std::endl;
  std::cout << "\nAvailable modes. A help for a mode: "
            << CommandLine::BinaryName() << " <mode> --help\n";
  for (const auto& mode : CommandLine::GetModes()) {
    std::cout << "  " << std::setw(10) << std::left << mode.first << " "
              << mode.second << std::endl;
  }
  std::cout << "\nAllowed command line flags for current mode:\n";
  std::cout << FormatFlag('h', "help", "Show help and exit.");
  std::cout << FormatFlag('\0', "show-hidden",
                          "Show hidden options. Use with --help.");
  for (const auto& option : options_) {
    if (!option->hidden_) std::cout << option->GetHelp(defaults_);
  }
  auto contexts = values_.ListSubdicts();
  if (!contexts.empty()) {
    std::cout << "\nFlags can be defined per context (one of: "
              << StrJoin(contexts, ", ") << "), for example:\n";
    std::cout << "       --" << contexts[0] << '.'
              << options_.back()->GetLongFlag() << "=(value)\n";
  }
}
void OptionsParser::ShowHidden() const {
  for (const auto& option : options_) option->hidden_ = false;
}
/////////////////////////////////////////////////////////////////
// StringOption
/////////////////////////////////////////////////////////////////
StringOption::StringOption(const OptionId& id) : Option(id) {}
void StringOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, value);
}
bool StringOption::ProcessLongFlag(const std::string& flag,
                                   const std::string& value,
                                   OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}
bool StringOption::ProcessShortFlagWithValue(char flag,
                                             const std::string& value,
                                             OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}
std::string StringOption::GetHelp(const OptionsDict& dict) const {
  return FormatFlag(GetShortFlag(), GetLongFlag() + "=STRING", GetHelpText(),
                    GetUciOption(), GetVal(dict));
}
std::string StringOption::GetOptionString(const OptionsDict& dict) const {
  return "type string default " + GetVal(dict);
}
std::string StringOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}
void StringOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetId(), val);
}
/////////////////////////////////////////////////////////////////
// IntOption
/////////////////////////////////////////////////////////////////
IntOption::IntOption(const OptionId& id, int min, int max)
    : Option(id), min_(min), max_(max) {}
void IntOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, ValidateIntString(value));
}
bool IntOption::ProcessLongFlag(const std::string& flag,
                                const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, ValidateIntString(value));
    return true;
  }
  return false;
}
bool IntOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                          OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, ValidateIntString(value));
    return true;
  }
  return false;
}
std::string IntOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag += "=" + std::to_string(min_) + ".." + std::to_string(max_);
  }
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    std::to_string(GetVal(dict)) +
                        "  MIN: " + std::to_string(min_) +
                        "  MAX: " + std::to_string(max_));
}
std::string IntOption::GetOptionString(const OptionsDict& dict) const {
  return "type spin default " + std::to_string(GetVal(dict)) + " min " +
         std::to_string(min_) + " max " + std::to_string(max_);
}
IntOption::ValueType IntOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}
void IntOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  if (val < min_ || val > max_) {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be between " << min_
        << " and " << max_ << ".";
    throw Exception(buf.str());
  }
  dict->Set<ValueType>(GetId(), val);
}
#ifndef NO_CHARCONV
int IntOption::ValidateIntString(const std::string& val) const {
  int result;
  const auto end = val.data() + val.size();
  auto [ptr, err] = std::from_chars(val.data(), end, result);
  if (err == std::errc::invalid_argument) {
    throw Exception("Flag '--" + GetLongFlag() + "' has an invalid format.");
  } else if (err == std::errc::result_out_of_range) {
    throw Exception("Flag '--" + GetLongFlag() + "' is out of range.");
  } else if (ptr != end) {
    throw Exception("Flag '--" + GetLongFlag() + "' has trailing characters.");
  } else {
    return result;
  }
}
#else
int IntOption::ValidateIntString(const std::string& val) const {
  char* end;
  errno = 0;
  int result = std::strtol(val.c_str(), &end, 10);
  if (errno == ERANGE) {
    throw Exception("Flag '--" + GetLongFlag() + "' is out of range.");
  } else if (val.length() == 0 || *end != '\0') {
    throw Exception("Flag '--" + GetLongFlag() + "' value is invalid.");
  } else {
    return result;
  }
}
#endif
/////////////////////////////////////////////////////////////////
// FloatOption
/////////////////////////////////////////////////////////////////
FloatOption::FloatOption(const OptionId& id, float min, float max)
    : Option(id), min_(min), max_(max) {}
void FloatOption::SetValue(const std::string& value, OptionsDict* dict) {
  try {
    SetVal(dict, std::stof(value));
  } catch (std::invalid_argument&) {
    throw Exception("invalid value " + value);
  } catch (const std::out_of_range&) {
    throw Exception("out of range value " + value);
  }
}
bool FloatOption::ProcessLongFlag(const std::string& flag,
                                  const std::string& value, OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    try {
      SetVal(dict, std::stof(value));
    } catch (std::invalid_argument&) {
      throw Exception("invalid value " + value);
    } catch (const std::out_of_range&) {
      throw Exception("out of range value " + value);
    }
    return true;
  }
  return false;
}
bool FloatOption::ProcessShortFlagWithValue(char flag, const std::string& value,
                                            OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    try {
      SetVal(dict, std::stof(value));
    } catch (std::invalid_argument&) {
      throw Exception("invalid value " + value);
    } catch (const std::out_of_range&) {
      throw Exception("out of range value " + value);
    }
    return true;
  }
  return false;
}
std::string FloatOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << min_ << ".." << max_;
    long_flag += "=" + oss.str();
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << GetVal(dict) << "  MIN: " << min_
      << "  MAX: " << max_;
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    oss.str());
}
std::string FloatOption::GetOptionString(const OptionsDict& dict) const {
  return "type string default " + std::to_string(GetVal(dict));
}
FloatOption::ValueType FloatOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}
void FloatOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  if (val < min_ || val > max_) {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be between " << min_
        << " and " << max_ << ".";
    throw Exception(buf.str());
  }
  dict->Set<ValueType>(GetId(), val);
}
/////////////////////////////////////////////////////////////////
// BoolOption
/////////////////////////////////////////////////////////////////
BoolOption::BoolOption(const OptionId& id) : Option(id) {}
void BoolOption::SetValue(const std::string& value, OptionsDict* dict) {
  ValidateBoolString(value);
  SetVal(dict, value == "true");
}
bool BoolOption::ProcessLongFlag(const std::string& flag,
                                 const std::string& value, OptionsDict* dict) {
  if (flag == "no-" + GetLongFlag()) {
    SetVal(dict, false);
    return true;
  }
  if (flag == GetLongFlag() && value.empty()) {
    SetVal(dict, true);
    return true;
  }
  ValidateBoolString(value);
  if (flag == GetLongFlag()) {
    SetVal(dict, value.empty() || (value != "false"));
    return true;
  }
  return false;
}
bool BoolOption::ProcessShortFlag(char flag, OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, !GetVal(*dict));
    return true;
  }
  return false;
}
std::string BoolOption::GetHelp(const OptionsDict& dict) const {
  std::string long_flag = GetLongFlag();
  if (!long_flag.empty()) {
    long_flag = "[no-]" + long_flag;
  }
  return FormatFlag(GetShortFlag(), long_flag, GetHelpText(), GetUciOption(),
                    GetVal(dict) ? "true" : "false");
}
std::string BoolOption::GetOptionString(const OptionsDict& dict) const {
  return "type check default " + std::string(GetVal(dict) ? "true" : "false");
}
BoolOption::ValueType BoolOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}
void BoolOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  dict->Set<ValueType>(GetId(), val);
}
void BoolOption::ValidateBoolString(const std::string& val) {
  if (val != "true" && val != "false") {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be either "
        << "'true' or 'false'.";
    throw Exception(buf.str());
  }
}
/////////////////////////////////////////////////////////////////
// ChoiceOption
/////////////////////////////////////////////////////////////////
ChoiceOption::ChoiceOption(const OptionId& id,
                           const std::vector<std::string>& choices)
    : Option(id), choices_(choices) {}
void ChoiceOption::SetValue(const std::string& value, OptionsDict* dict) {
  SetVal(dict, value);
}
bool ChoiceOption::ProcessLongFlag(const std::string& flag,
                                   const std::string& value,
                                   OptionsDict* dict) {
  if (flag == GetLongFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}
bool ChoiceOption::ProcessShortFlagWithValue(char flag,
                                             const std::string& value,
                                             OptionsDict* dict) {
  if (flag == GetShortFlag()) {
    SetVal(dict, value);
    return true;
  }
  return false;
}
std::string ChoiceOption::GetHelp(const OptionsDict& dict) const {
  std::string values;
  for (const auto& choice : choices_) {
    if (!values.empty()) values += ',';
    values += choice;
  }
  return FormatFlag(GetShortFlag(), GetLongFlag() + "=CHOICE", GetHelpText(),
                    GetUciOption(), GetVal(dict) + "  VALUES: " + values);
}
std::string ChoiceOption::GetOptionString(const OptionsDict& dict) const {
  std::string res = "type combo default " + GetVal(dict);
  for (const auto& choice : choices_) {
    res += " var " + choice;
  }
  return res;
}
std::string ChoiceOption::GetVal(const OptionsDict& dict) const {
  return dict.Get<ValueType>(GetId());
}
void ChoiceOption::SetVal(OptionsDict* dict, const ValueType& val) const {
  bool valid = false;
  std::string choice_string;
  for (const auto& choice : choices_) {
    choice_string += " " + choice;
    if (val == choice) {
      valid = true;
      break;
    }
  }
  if (!valid) {
    std::ostringstream buf;
    buf << "Flag '--" << GetLongFlag() << "' must be one of the "
        << "following values:" << choice_string << ".";
    throw Exception(buf.str());
  }
  dict->Set<ValueType>(GetId(), val);
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/optionsparser.cc

// begin of /Users/syys/CLionProjects/lc0/src/mcts/params.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#if __has_include("params_override.h")
#endif
#ifndef DEFAULT_MINIBATCH_SIZE
#define DEFAULT_MINIBATCH_SIZE 256
#endif
#ifndef DEFAULT_MAX_PREFETCH
#define DEFAULT_MAX_PREFETCH 32
#endif
#ifndef DEFAULT_TASK_WORKERS
#define DEFAULT_TASK_WORKERS 4
#endif
namespace lczero {
namespace {
FillEmptyHistory EncodeHistoryFill(std::string history_fill) {
  if (history_fill == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (history_fill == "always") return FillEmptyHistory::ALWAYS;
  assert(history_fill == "no");
  return FillEmptyHistory::NO;
}
}  // namespace
const OptionId SearchParams::kMiniBatchSizeId{
    "minibatch-size", "MinibatchSize",
    "How many positions the engine tries to batch together for parallel NN "
    "computation. Larger batches may reduce strength a bit, especially with a "
    "small number of playouts."};
const OptionId SearchParams::kMaxPrefetchBatchId{
    "max-prefetch", "MaxPrefetch",
    "When the engine cannot gather a large enough batch for immediate use, try "
    "to prefetch up to X positions which are likely to be useful soon, and put "
    "them into cache."};
const OptionId SearchParams::kCpuctId{
    "cpuct", "CPuct",
    "cpuct_init constant from \"UCT search\" algorithm. Higher values promote "
    "more exploration/wider search, lower values promote more "
    "confidence/deeper search."};
const OptionId SearchParams::kCpuctAtRootId{
    "cpuct-at-root", "CPuctAtRoot",
    "cpuct_init constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctBaseId{
    "cpuct-base", "CPuctBase",
    "cpuct_base constant from \"UCT search\" algorithm. Lower value means "
    "higher growth of Cpuct as number of node visits grows."};
const OptionId SearchParams::kCpuctBaseAtRootId{
    "cpuct-base-at-root", "CPuctBaseAtRoot",
    "cpuct_base constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctFactorId{
    "cpuct-factor", "CPuctFactor", "Multiplier for the cpuct growth formula."};
const OptionId SearchParams::kCpuctFactorAtRootId{
    "cpuct-factor-at-root", "CPuctFactorAtRoot",
    "Multiplier for the cpuct growth formula at root."};
// Remove this option after 0.25 has been made mandatory in training and the
// training server stops sending it.
const OptionId SearchParams::kRootHasOwnCpuctParamsId{
    "root-has-own-cpuct-params", "RootHasOwnCpuctParams",
    "If enabled, cpuct parameters for root node are taken from *AtRoot "
    "parameters. Otherwise, they are the same as for the rest of nodes. "
    "Temporary flag for transition to a new version."};
const OptionId SearchParams::kTwoFoldDrawsId{
    "two-fold-draws", "TwoFoldDraws",
    "Evaluates twofold repetitions in the search tree as draws. Visits to "
    "these positions are reverted when the first occurrence is played "
    "and not in the search tree anymore."};
const OptionId SearchParams::kTemperatureId{
    "temperature", "Temperature",
    "Tau value from softmax formula for the first move. If equal to 0, the "
    "engine picks the best move to make. Larger values increase randomness "
    "while making the move."};
const OptionId SearchParams::kTempDecayMovesId{
    "tempdecay-moves", "TempDecayMoves",
    "Reduce temperature for every move after the first move, decreasing "
    "linearly over this number of moves from initial temperature to 0. "
    "A value of 0 disables tempdecay."};
const OptionId SearchParams::kTempDecayDelayMovesId{
    "tempdecay-delay-moves", "TempDecayDelayMoves",
    "Delay the linear decrease of temperature by this number of moves, "
    "decreasing linearly from initial temperature to 0. A value of 0 starts "
    "tempdecay after the first move."};
const OptionId SearchParams::kTemperatureCutoffMoveId{
    "temp-cutoff-move", "TempCutoffMove",
    "Move number, starting from which endgame temperature is used rather "
    "than initial temperature. Setting it to 0 disables cutoff."};
const OptionId SearchParams::kTemperatureEndgameId{
    "temp-endgame", "TempEndgame",
    "Temperature used during endgame (starting from cutoff move). Endgame "
    "temperature doesn't decay."};
const OptionId SearchParams::kTemperatureWinpctCutoffId{
    "temp-value-cutoff", "TempValueCutoff",
    "When move is selected using temperature, bad moves (with win "
    "probability less than X than the best move) are not considered at all."};
const OptionId SearchParams::kTemperatureVisitOffsetId{
    "temp-visit-offset", "TempVisitOffset",
    "Adjusts visits by this value when picking a move with a temperature. If a "
    "negative offset reduces visits for a particular move below zero, that "
    "move is not picked. If no moves can be picked, no temperature is used."};
const OptionId SearchParams::kNoiseEpsilonId{
    "noise-epsilon", "DirichletNoiseEpsilon",
    "Amount of Dirichlet noise to combine with root priors. This allows the "
    "engine to discover new ideas during training by exploring moves which are "
    "known to be bad. Not normally used during play."};
const OptionId SearchParams::kNoiseAlphaId{
    "noise-alpha", "DirichletNoiseAlpha",
    "Alpha of Dirichlet noise to control the sharpness of move probabilities. "
    "Larger values result in flatter / more evenly distributed values."};
const OptionId SearchParams::kVerboseStatsId{
    "verbose-move-stats", "VerboseMoveStats",
    "Display Q, V, N, U and P values of every move candidate after each move.",
    'v'};
const OptionId SearchParams::kLogLiveStatsId{
    "log-live-stats", "LogLiveStats",
    "Do VerboseMoveStats on every info update."};
const OptionId SearchParams::kFpuStrategyId{
    "fpu-strategy", "FpuStrategy",
    "How is an eval of unvisited node determined. \"First Play Urgency\" "
    "changes search behavior to visit unvisited nodes earlier or later by "
    "using a placeholder eval before checking the network. The value specified "
    "with --fpu-value results in \"reduction\" subtracting that value from the "
    "parent eval while \"absolute\" directly uses that value."};
const OptionId SearchParams::kFpuValueId{
    "fpu-value", "FpuValue",
    "\"First Play Urgency\" value used to adjust unvisited node eval based on "
    "--fpu-strategy."};
const OptionId SearchParams::kFpuStrategyAtRootId{
    "fpu-strategy-at-root", "FpuStrategyAtRoot",
    "How is an eval of unvisited root children determined. Just like "
    "--fpu-strategy except only at the root level and adjusts unvisited root "
    "children eval with --fpu-value-at-root. In addition to matching the "
    "strategies from --fpu-strategy, this can be \"same\" to disable the "
    "special root behavior."};
const OptionId SearchParams::kFpuValueAtRootId{
    "fpu-value-at-root", "FpuValueAtRoot",
    "\"First Play Urgency\" value used to adjust unvisited root children eval "
    "based on --fpu-strategy-at-root. Has no effect if --fpu-strategy-at-root "
    "is \"same\"."};
const OptionId SearchParams::kCacheHistoryLengthId{
    "cache-history-length", "CacheHistoryLength",
    "Length of history, in half-moves, to include into the cache key. When "
    "this value is less than history that NN uses to eval a position, it's "
    "possble that the search will use eval of the same position with different "
    "history taken from cache."};
const OptionId SearchParams::kPolicySoftmaxTempId{
    "policy-softmax-temp", "PolicyTemperature",
    "Policy softmax temperature. Higher values make priors of move candidates "
    "closer to each other, widening the search."};
const OptionId SearchParams::kMaxCollisionVisitsId{
    "max-collision-visits", "MaxCollisionVisits",
    "Total allowed node collision visits, per batch."};
const OptionId SearchParams::kMaxCollisionEventsId{
    "max-collision-events", "MaxCollisionEvents",
    "Allowed node collision events, per batch."};
const OptionId SearchParams::kOutOfOrderEvalId{
    "out-of-order-eval", "OutOfOrderEval",
    "During the gathering of a batch for NN to eval, if position happens to be "
    "in the cache or is terminal, evaluate it right away without sending the "
    "batch to the NN. When off, this may only happen with the very first node "
    "of a batch; when on, this can happen with any node."};
const OptionId SearchParams::kMaxOutOfOrderEvalsId{
    "max-out-of-order-evals-factor", "MaxOutOfOrderEvalsFactor",
    "Maximum number of out of order evals during gathering of a batch is "
    "calculated by multiplying the maximum batch size by this number."};
const OptionId SearchParams::kStickyEndgamesId{
    "sticky-endgames", "StickyEndgames",
    "When an end of game position is found during search, allow the eval of "
    "the previous move's position to stick to something more accurate. For "
    "example, if at least one move results in checkmate, then the position "
    "should stick as checkmated. Similarly, if all moves are drawn or "
    "checkmated, the position should stick as drawn or checkmate."};
const OptionId SearchParams::kSyzygyFastPlayId{
    "syzygy-fast-play", "SyzygyFastPlay",
    "With DTZ tablebase files, only allow the network pick from winning moves "
    "that have shortest DTZ to play faster (but not necessarily optimally)."};
const OptionId SearchParams::kMultiPvId{
    "multipv", "MultiPV",
    "Number of game play lines (principal variations) to show in UCI info "
    "output."};
const OptionId SearchParams::kPerPvCountersId{
    "per-pv-counters", "PerPVCounters",
    "Show node counts per principal variation instead of total nodes in UCI."};
const OptionId SearchParams::kScoreTypeId{
    "score-type", "ScoreType",
    "What to display as score. Either centipawns (the UCI default), win "
    "percentage or Q (the actual internal score) multiplied by 100."};
const OptionId SearchParams::kHistoryFillId{
    "history-fill", "HistoryFill",
    "Neural network uses 7 previous board positions in addition to the current "
    "one. During the first moves of the game such historical positions don't "
    "exist, but they can be synthesized. This parameter defines when to "
    "synthesize them (always, never, or only at non-standard fen position)."};
const OptionId SearchParams::kMovesLeftMaxEffectId{
    "moves-left-max-effect", "MovesLeftMaxEffect",
    "Maximum bonus to add to the score of a node based on how much "
    "shorter/longer it makes the game when winning/losing."};
const OptionId SearchParams::kMovesLeftThresholdId{
    "moves-left-threshold", "MovesLeftThreshold",
    "Absolute value of node Q needs to exceed this value before shorter wins "
    "or longer losses are considered."};
const OptionId SearchParams::kMovesLeftSlopeId{
    "moves-left-slope", "MovesLeftSlope",
    "Controls how the bonus for shorter wins or longer losses is adjusted "
    "based on how many moves the move is estimated to shorten/lengthen the "
    "game. The move difference is multiplied with the slope and capped at "
    "MovesLeftMaxEffect."};
const OptionId SearchParams::kMovesLeftConstantFactorId{
    "moves-left-constant-factor", "MovesLeftConstantFactor",
    "A simple multiplier to the moves left effect, can be set to 0 to only use "
    "an effect scaled by Q."};
const OptionId SearchParams::kMovesLeftScaledFactorId{
    "moves-left-scaled-factor", "MovesLeftScaledFactor",
    "A factor which is multiplied by the absolute Q of parent node and the "
    "base moves left effect."};
const OptionId SearchParams::kMovesLeftQuadraticFactorId{
    "moves-left-quadratic-factor", "MovesLeftQuadraticFactor",
    "A factor which is multiplied by the square of Q of parent node and the "
    "base moves left effect."};
const OptionId SearchParams::kDisplayCacheUsageId{
    "display-cache-usage", "DisplayCacheUsage",
    "Display cache fullness through UCI info `hash` section."};
const OptionId SearchParams::kMaxConcurrentSearchersId{
    "max-concurrent-searchers", "MaxConcurrentSearchers",
    "If not 0, at most this many search workers can be gathering minibatches "
    "at once."};
const OptionId SearchParams::kDrawScoreSidetomoveId{
    "draw-score-sidetomove", "DrawScoreSideToMove",
    "Score of a drawn game, as seen by a player making the move."};
const OptionId SearchParams::kDrawScoreOpponentId{
    "draw-score-opponent", "DrawScoreOpponent",
    "Score of a drawn game, as seen by the opponent."};
const OptionId SearchParams::kDrawScoreWhiteId{
    "draw-score-white", "DrawScoreWhite",
    "Adjustment, added to a draw score of a white player."};
const OptionId SearchParams::kDrawScoreBlackId{
    "draw-score-black", "DrawScoreBlack",
    "Adjustment, added to a draw score of a black player."};
const OptionId SearchParams::kNpsLimitId{
    "nps-limit", "NodesPerSecondLimit",
    "An option to specify an upper limit to the nodes per second searched. The "
    "accuracy depends on the minibatch size used, increasing for lower sizes, "
    "and on the length of the search. Zero to disable."};
const OptionId SearchParams::kSolidTreeThresholdId{
    "solid-tree-threshold", "SolidTreeThreshold",
    "Only nodes with at least this number of visits will be considered for "
    "solidification for improved cache locality."};
const OptionId SearchParams::kTaskWorkersPerSearchWorkerId{
    "task-workers", "TaskWorkers",
    "The number of task workers to use to help the search worker."};
const OptionId SearchParams::kMinimumWorkSizeForProcessingId{
    "minimum-processing-work", "MinimumProcessingWork",
    "This many visits need to be gathered before tasks will be used to "
    "accelerate processing."};
const OptionId SearchParams::kMinimumWorkSizeForPickingId{
    "minimum-picking-work", "MinimumPickingWork",
    "Search branches with more than this many collisions/visits may be split "
    "off to task workers."};
const OptionId SearchParams::kMinimumRemainingWorkSizeForPickingId{
    "minimum-remaining-picking-work", "MinimumRemainingPickingWork",
    "Search branches won't be split off to task workers unless there is at "
    "least this much work left to do afterwards."};
const OptionId SearchParams::kMinimumWorkPerTaskForProcessingId{
    "minimum-per-task-processing", "MinimumPerTaskProcessing",
    "Processing work won't be split into chunks smaller than this (unless its "
    "more than half of MinimumProcessingWork)."};
const OptionId SearchParams::kIdlingMinimumWorkId{
    "idling-minimum-work", "IdlingMinimumWork",
    "Only early exit gathering due to 'idle' backend if more than this many "
    "nodes will be sent to the backend."};
const OptionId SearchParams::kThreadIdlingThresholdId{
    "thread-idling-threshold", "ThreadIdlingThreshold",
    "If there are more than this number of search threads that are not "
    "actively in the process of either sending data to the backend or waiting "
    "for data from the backend, assume that the backend is idle."};
const OptionId SearchParams::kMaxCollisionVisitsScalingStartId{
    "max-collision-visits-scaling-start", "MaxCollisionVisitsScalingStart",
    "Tree size where max collision visits starts scaling up from 1."};
const OptionId SearchParams::kMaxCollisionVisitsScalingEndId{
    "max-collision-visits-scaling-end", "MaxCollisionVisitsScalingEnd",
    "Tree size where max collision visits reaches max. Set to 0 to disable "
    "scaling entirely."};
const OptionId SearchParams::kMaxCollisionVisitsScalingPowerId{
    "max-collision-visits-scaling-power", "MaxCollisionVisitsScalingPower",
    "Power to apply to the interpolation between 1 and max to make it curved."};
void SearchParams::Populate(OptionsParser* options) {
  // Here the uci optimized defaults" are set.
  // Many of them are overridden with training specific values in tournament.cc.
  options->Add<IntOption>(kMiniBatchSizeId, 1, 1024) = DEFAULT_MINIBATCH_SIZE;
  options->Add<IntOption>(kMaxPrefetchBatchId, 0, 1024) = DEFAULT_MAX_PREFETCH;
  options->Add<FloatOption>(kCpuctId, 0.0f, 100.0f) = 1.745f;
  options->Add<FloatOption>(kCpuctAtRootId, 0.0f, 100.0f) = 1.745f;
  options->Add<FloatOption>(kCpuctBaseId, 1.0f, 1000000000.0f) = 38739.0f;
  options->Add<FloatOption>(kCpuctBaseAtRootId, 1.0f, 1000000000.0f) = 38739.0f;
  options->Add<FloatOption>(kCpuctFactorId, 0.0f, 1000.0f) = 3.894f;
  options->Add<FloatOption>(kCpuctFactorAtRootId, 0.0f, 1000.0f) = 3.894f;
  options->Add<BoolOption>(kRootHasOwnCpuctParamsId) = false;
  options->Add<BoolOption>(kTwoFoldDrawsId) = true;
  options->Add<FloatOption>(kTemperatureId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kTempDecayMovesId, 0, 640) = 0;
  options->Add<IntOption>(kTempDecayDelayMovesId, 0, 100) = 0;
  options->Add<IntOption>(kTemperatureCutoffMoveId, 0, 1000) = 0;
  options->Add<FloatOption>(kTemperatureEndgameId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kTemperatureWinpctCutoffId, 0.0f, 100.0f) = 100.0f;
  options->Add<FloatOption>(kTemperatureVisitOffsetId, -1000.0f, 1000.0f) =
      0.0f;
  options->Add<FloatOption>(kNoiseEpsilonId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kNoiseAlphaId, 0.0f, 10000000.0f) = 0.3f;
  options->Add<BoolOption>(kVerboseStatsId) = false;
  options->Add<BoolOption>(kLogLiveStatsId) = false;
  std::vector<std::string> fpu_strategy = {"reduction", "absolute"};
  options->Add<ChoiceOption>(kFpuStrategyId, fpu_strategy) = "reduction";
  options->Add<FloatOption>(kFpuValueId, -100.0f, 100.0f) = 0.330f;
  fpu_strategy.push_back("same");
  options->Add<ChoiceOption>(kFpuStrategyAtRootId, fpu_strategy) = "same";
  options->Add<FloatOption>(kFpuValueAtRootId, -100.0f, 100.0f) = 1.0f;
  options->Add<IntOption>(kCacheHistoryLengthId, 0, 7) = 0;
  options->Add<FloatOption>(kPolicySoftmaxTempId, 0.1f, 10.0f) = 1.359f;
  options->Add<IntOption>(kMaxCollisionEventsId, 1, 65536) = 917;
  options->Add<IntOption>(kMaxCollisionVisitsId, 1, 100000000) = 80000;
  options->Add<IntOption>(kMaxCollisionVisitsScalingStartId, 1, 100000) = 28;
  options->Add<IntOption>(kMaxCollisionVisitsScalingEndId, 0, 100000000) =
      145000;
  options->Add<FloatOption>(kMaxCollisionVisitsScalingPowerId, 0.01, 100) =
      1.25;
  options->Add<BoolOption>(kOutOfOrderEvalId) = true;
  options->Add<FloatOption>(kMaxOutOfOrderEvalsId, 0.0f, 100.0f) = 2.4f;
  options->Add<BoolOption>(kStickyEndgamesId) = true;
  options->Add<BoolOption>(kSyzygyFastPlayId) = false;
  options->Add<IntOption>(kMultiPvId, 1, 500) = 1;
  options->Add<BoolOption>(kPerPvCountersId) = false;
  std::vector<std::string> score_type = {"centipawn",
                                         "centipawn_with_drawscore",
                                         "centipawn_2019",
                                         "centipawn_2018",
                                         "win_percentage",
                                         "Q",
                                         "W-L"};
  options->Add<ChoiceOption>(kScoreTypeId, score_type) = "centipawn";
  std::vector<std::string> history_fill_opt{"no", "fen_only", "always"};
  options->Add<ChoiceOption>(kHistoryFillId, history_fill_opt) = "fen_only";
  options->Add<FloatOption>(kMovesLeftMaxEffectId, 0.0f, 1.0f) = 0.0345f;
  options->Add<FloatOption>(kMovesLeftThresholdId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kMovesLeftSlopeId, 0.0f, 1.0f) = 0.0027f;
  options->Add<FloatOption>(kMovesLeftConstantFactorId, -1.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kMovesLeftScaledFactorId, -2.0f, 2.0f) = 1.6521f;
  options->Add<FloatOption>(kMovesLeftQuadraticFactorId, -1.0f, 1.0f) =
      -0.6521f;
  options->Add<BoolOption>(kDisplayCacheUsageId) = false;
  options->Add<IntOption>(kMaxConcurrentSearchersId, 0, 128) = 1;
  options->Add<IntOption>(kDrawScoreSidetomoveId, -100, 100) = 0;
  options->Add<IntOption>(kDrawScoreOpponentId, -100, 100) = 0;
  options->Add<IntOption>(kDrawScoreWhiteId, -100, 100) = 0;
  options->Add<IntOption>(kDrawScoreBlackId, -100, 100) = 0;
  options->Add<FloatOption>(kNpsLimitId, 0.0f, 1e6f) = 0.0f;
  options->Add<IntOption>(kSolidTreeThresholdId, 1, 2000000000) = 100;
  options->Add<IntOption>(kTaskWorkersPerSearchWorkerId, 0, 128) =
      DEFAULT_TASK_WORKERS;
  options->Add<IntOption>(kMinimumWorkSizeForProcessingId, 2, 100000) = 20;
  options->Add<IntOption>(kMinimumWorkSizeForPickingId, 1, 100000) = 1;
  options->Add<IntOption>(kMinimumRemainingWorkSizeForPickingId, 0, 100000) =
      20;
  options->Add<IntOption>(kMinimumWorkPerTaskForProcessingId, 1, 100000) = 8;
  options->Add<IntOption>(kIdlingMinimumWorkId, 0, 10000) = 0;
  options->Add<IntOption>(kThreadIdlingThresholdId, 0, 128) = 1;
  options->HideOption(kNoiseEpsilonId);
  options->HideOption(kNoiseAlphaId);
  options->HideOption(kLogLiveStatsId);
  options->HideOption(kDisplayCacheUsageId);
  options->HideOption(kRootHasOwnCpuctParamsId);
  options->HideOption(kCpuctAtRootId);
  options->HideOption(kCpuctBaseAtRootId);
  options->HideOption(kCpuctFactorAtRootId);
  options->HideOption(kFpuStrategyAtRootId);
  options->HideOption(kFpuValueAtRootId);
  options->HideOption(kTemperatureId);
  options->HideOption(kTempDecayMovesId);
  options->HideOption(kTempDecayDelayMovesId);
  options->HideOption(kTemperatureCutoffMoveId);
  options->HideOption(kTemperatureEndgameId);
  options->HideOption(kTemperatureWinpctCutoffId);
  options->HideOption(kTemperatureVisitOffsetId);
}
SearchParams::SearchParams(const OptionsDict& options)
    : options_(options),
      kCpuct(options.Get<float>(kCpuctId)),
      kCpuctAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctAtRootId
                                                      : kCpuctId)),
      kCpuctBase(options.Get<float>(kCpuctBaseId)),
      kCpuctBaseAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctBaseAtRootId
                                                      : kCpuctBaseId)),
      kCpuctFactor(options.Get<float>(kCpuctFactorId)),
      kCpuctFactorAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctFactorAtRootId
                                                      : kCpuctFactorId)),
      kTwoFoldDraws(options.Get<bool>(kTwoFoldDrawsId)),
      kNoiseEpsilon(options.Get<float>(kNoiseEpsilonId)),
      kNoiseAlpha(options.Get<float>(kNoiseAlphaId)),
      kFpuAbsolute(options.Get<std::string>(kFpuStrategyId) == "absolute"),
      kFpuValue(options.Get<float>(kFpuValueId)),
      kFpuAbsoluteAtRoot(
          (options.Get<std::string>(kFpuStrategyAtRootId) == "same" &&
           kFpuAbsolute) ||
          options.Get<std::string>(kFpuStrategyAtRootId) == "absolute"),
      kFpuValueAtRoot(options.Get<std::string>(kFpuStrategyAtRootId) == "same"
                          ? kFpuValue
                          : options.Get<float>(kFpuValueAtRootId)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthId)),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempId)),
      kMaxCollisionEvents(options.Get<int>(kMaxCollisionEventsId)),
      kMaxCollisionVisits(options.Get<int>(kMaxCollisionVisitsId)),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalId)),
      kStickyEndgames(options.Get<bool>(kStickyEndgamesId)),
      kSyzygyFastPlay(options.Get<bool>(kSyzygyFastPlayId)),
      kHistoryFill(EncodeHistoryFill(options.Get<std::string>(kHistoryFillId))),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeId)),
      kMovesLeftMaxEffect(options.Get<float>(kMovesLeftMaxEffectId)),
      kMovesLeftThreshold(options.Get<float>(kMovesLeftThresholdId)),
      kMovesLeftSlope(options.Get<float>(kMovesLeftSlopeId)),
      kMovesLeftConstantFactor(options.Get<float>(kMovesLeftConstantFactorId)),
      kMovesLeftScaledFactor(options.Get<float>(kMovesLeftScaledFactorId)),
      kMovesLeftQuadraticFactor(
          options.Get<float>(kMovesLeftQuadraticFactorId)),
      kDisplayCacheUsage(options.Get<bool>(kDisplayCacheUsageId)),
      kMaxConcurrentSearchers(options.Get<int>(kMaxConcurrentSearchersId)),
      kDrawScoreSidetomove{options.Get<int>(kDrawScoreSidetomoveId) / 100.0f},
      kDrawScoreOpponent{options.Get<int>(kDrawScoreOpponentId) / 100.0f},
      kDrawScoreWhite{options.Get<int>(kDrawScoreWhiteId) / 100.0f},
      kDrawScoreBlack{options.Get<int>(kDrawScoreBlackId) / 100.0f},
      kMaxOutOfOrderEvals(std::max(
          1, static_cast<int>(options.Get<float>(kMaxOutOfOrderEvalsId) *
                              options.Get<int>(kMiniBatchSizeId)))),
      kNpsLimit(options.Get<float>(kNpsLimitId)),
      kSolidTreeThreshold(options.Get<int>(kSolidTreeThresholdId)),
      kTaskWorkersPerSearchWorker(options.Get<int>(kTaskWorkersPerSearchWorkerId)),
      kMinimumWorkSizeForProcessing(
          options.Get<int>(kMinimumWorkSizeForProcessingId)),
      kMinimumWorkSizeForPicking(
          options.Get<int>(kMinimumWorkSizeForPickingId)),
      kMinimumRemainingWorkSizeForPicking(
          options.Get<int>(kMinimumRemainingWorkSizeForPickingId)),
      kMinimumWorkPerTaskForProcessing(
          options.Get<int>(kMinimumWorkPerTaskForProcessingId)),
      kIdlingMinimumWork(options.Get<int>(kIdlingMinimumWorkId)),
      kThreadIdlingThreshold(options.Get<int>(kThreadIdlingThresholdId)),
      kMaxCollisionVisitsScalingStart(
          options.Get<int>(kMaxCollisionVisitsScalingStartId)),
      kMaxCollisionVisitsScalingEnd(
          options.Get<int>(kMaxCollisionVisitsScalingEndId)),
      kMaxCollisionVisitsScalingPower(
          options.Get<float>(kMaxCollisionVisitsScalingPowerId)) {
  if (std::max(std::abs(kDrawScoreSidetomove), std::abs(kDrawScoreOpponent)) +
          std::max(std::abs(kDrawScoreWhite), std::abs(kDrawScoreBlack)) >
      1.0f) {
    throw Exception(
        "max{|sidetomove|+|opponent|} + max{|white|+|black|} draw score must "
        "be <= 100");
  }
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/params.cc

// begin of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/timemgr.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
StoppersHints::StoppersHints() { Reset(); }
void StoppersHints::UpdateEstimatedRemainingTimeMs(int64_t v) {
  if (v < remaining_time_ms_) remaining_time_ms_ = v;
}
int64_t StoppersHints::GetEstimatedRemainingTimeMs() const {
  return remaining_time_ms_;
}
void StoppersHints::UpdateEstimatedRemainingPlayouts(int64_t v) {
  if (v < remaining_playouts_) remaining_playouts_ = v;
}
int64_t StoppersHints::GetEstimatedRemainingPlayouts() const {
  // Even if we exceeded limits, don't go crazy by not allowing any playouts.
  return std::max(decltype(remaining_playouts_){1}, remaining_playouts_);
}
void StoppersHints::UpdateEstimatedNps(float v) { estimated_nps_ = v; }
std::optional<float> StoppersHints::GetEstimatedNps() const {
  return estimated_nps_;
}
void StoppersHints::Reset() {
  // Slightly more than 3 years.
  remaining_time_ms_ = 100000000000;
  // Type for N in nodes is currently uint32_t, so set limit in order not to
  // overflow it.
  remaining_playouts_ = 4000000000;
  // NPS is not known.
  estimated_nps_.reset();
}
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/timemgr.cc

// begin of /Users/syys/CLionProjects/lc0/src/syzygy/syzygy.cc
/*
  Originally from cfish's tbprobe.c
  Copyright (c) 2013-2018 Ronald de Man
  That file may be redistributed and/or modified without restrictions.
  This modified version is available under the GPL:
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#ifndef _WIN32
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif
namespace lczero {
namespace {
#define TB_PIECES 7
#define TB_HASHBITS (TB_PIECES < 7 ? 11 : 12)
#define TB_MAX_PIECE (TB_PIECES < 7 ? 254 : 650)
#define TB_MAX_PAWN (TB_PIECES < 7 ? 256 : 861)
#ifdef _WIN32
typedef HANDLE map_t;
#define SEP_CHAR ';'
#else
typedef size_t map_t;
#define SEP_CHAR ':'
#endif
typedef uint64_t Key;
constexpr const char* kSuffix[] = {".rtbw", ".rtbm", ".rtbz"};
constexpr uint32_t kMagic[] = {0x5d23e871, 0x88ac504b, 0xa50c66d7};
enum { WDL, DTM, DTZ };
enum { PIECE_ENC, FILE_ENC, RANK_ENC };
enum PieceType {
  PAWN = 1,
  KNIGHT,
  BISHOP,
  ROOK,
  QUEEN,
  KING,
};
enum Piece {
  W_PAWN = 1,
  W_KNIGHT,
  W_BISHOP,
  W_ROOK,
  W_QUEEN,
  W_KING,
  B_PAWN = 9,
  B_KNIGHT,
  B_BISHOP,
  B_ROOK,
  B_QUEEN,
  B_KING,
};
struct PairsData {
  uint8_t* indexTable;
  uint16_t* sizeTable;
  uint8_t* data;
  uint16_t* offset;
  uint8_t* symLen;
  uint8_t* symPat;
  uint8_t blockSize;
  uint8_t idxBits;
  uint8_t minLen;
  uint8_t constValue[2];
  uint64_t base[1];  // must be base[1] in C++
};
struct EncInfo {
  PairsData* precomp;
  size_t factor[TB_PIECES];
  uint8_t pieces[TB_PIECES];
  uint8_t norm[TB_PIECES];
};
struct BaseEntry {
  Key key;
  uint8_t* data[3];
  map_t mapping[3];
  std::atomic<bool> ready[3];
  uint8_t num;
  bool symmetric;
  bool hasPawns;
  bool hasDtm;
  bool hasDtz;
  union {
    bool kk_enc;
    uint8_t pawns[2];
  };
  bool dtmLossOnly;
};
struct PieceEntry : BaseEntry {
  EncInfo ei[5];  // 2 + 2 + 1
  uint16_t* dtmMap;
  uint16_t dtmMapIdx[2][2];
  void* dtzMap;
  uint16_t dtzMapIdx[4];
  uint8_t dtzFlags;
};
struct PawnEntry : BaseEntry {
  EncInfo ei[24];  // 4 * 2 + 6 * 2 + 4
  uint16_t* dtmMap;
  uint16_t dtmMapIdx[6][2][2];
  void* dtzMap;
  uint16_t dtzMapIdx[4][4];
  uint8_t dtzFlags[4];
  bool dtmSwitched;
};
struct TbHashEntry {
  Key key;
  BaseEntry* ptr;
};
constexpr int kWdlToDtz[] = {-1, -101, 0, 101, 1};
// DTZ tables don't store valid scores for moves that reset the rule50 counter
// like captures and pawn moves but we can easily recover the correct dtz of the
// previous move if we know the position's WDL score.
int dtz_before_zeroing(WDLScore wdl) { return kWdlToDtz[wdl + 2]; }
// Return the sign of a number (-1, 0, 1)
template <typename T>
int sign_of(T val) {
  return (T(0) < val) - (val < T(0));
}
int count_pieces(const ChessBoard& pos, int type, bool theirs) {
  const BitBoard all = theirs ? pos.theirs() : pos.ours();
  switch (type) {
    case KING:
      return 1;
    case QUEEN:
      return (all & pos.queens()).count_few();
    case ROOK:
      return (all & pos.rooks()).count_few();
    case BISHOP:
      return (all & pos.bishops()).count_few();
    case KNIGHT:
      return (all & pos.knights()).count_few();
    case PAWN:
      return (all & pos.pawns()).count_few();
    default:
      assert(false);
  }
  return 0;
}
BitBoard pieces(const ChessBoard& pos, int type, bool theirs) {
  const BitBoard all = theirs ? pos.theirs() : pos.ours();
  switch (type) {
    case KING:
      return all & pos.kings();
    case QUEEN:
      return all & pos.queens();
    case ROOK:
      return all & pos.rooks();
    case BISHOP:
      return all & pos.bishops();
    case KNIGHT:
      return all & pos.knights();
    case PAWN:
      return all & pos.pawns();
    default:
      assert(false);
  }
  return BitBoard();
}
bool is_capture(const ChessBoard& pos, const Move& move) {
  // Simple capture.
  if (pos.theirs().get(move.to())) return true;
  // Enpassant capture. Pawn moves other than straight it must be a capture.
  if (pos.pawns().get(move.from()) && move.from().col() != move.to().col()) {
    return true;
  }
  return false;
}
constexpr char kPieceToChar[] = " PNBRQK  pnbrqk";
// Given a position, produce a text string of the form KQPvKRP, where
// "KQP" represents the white pieces if flip == false and the black pieces
// if flip == true.
void prt_str(const ChessBoard& pos, char* str, bool flip) {
  const bool first_theirs = flip ^ pos.flipped();
  for (int pt = KING; pt >= PAWN; pt--) {
    for (int i = count_pieces(pos, pt, first_theirs); i > 0; i--) {
      *str++ = kPieceToChar[pt];
    }
  }
  *str++ = 'v';
  for (int pt = KING; pt >= PAWN; pt--) {
    for (int i = count_pieces(pos, pt, !first_theirs); i > 0; i--) {
      *str++ = kPieceToChar[pt];
    }
  }
  *str++ = 0;
}
#define pchr(i) kPieceToChar[QUEEN - (i)]
#define Swap(a, b) \
  {                \
    int tmp = a;   \
    a = b;         \
    b = tmp;       \
  }
#define PIECE(x) (static_cast<PieceEntry*>(x))
#define PAWN(x) (static_cast<PawnEntry*>(x))
int num_tables(BaseEntry* be, const int type) {
  return be->hasPawns ? type == DTM ? 6 : 4 : 1;
}
EncInfo* first_ei(BaseEntry* be, const int type) {
  return be->hasPawns ? &PAWN(be)->ei[type == WDL ? 0 : type == DTM ? 8 : 20]
                      : &PIECE(be)->ei[type == WDL ? 0 : type == DTM ? 2 : 4];
}
constexpr int8_t kOffDiag[] = {
    0, -1, -1, -1, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1,
    1, 1,  0,  -1, -1, -1, -1, -1, 1, 1, 1,  0,  -1, -1, -1, -1,
    1, 1,  1,  1,  0,  -1, -1, -1, 1, 1, 1,  1,  1,  0,  -1, -1,
    1, 1,  1,  1,  1,  1,  0,  -1, 1, 1, 1,  1,  1,  1,  1,  0};
constexpr uint8_t kTriangle[] = {
    6, 0, 1, 2, 2, 1, 0, 6, 0, 7, 3, 4, 4, 3, 7, 0, 1, 3, 8, 5, 5, 8,
    3, 1, 2, 4, 5, 9, 9, 5, 4, 2, 2, 4, 5, 9, 9, 5, 4, 2, 1, 3, 8, 5,
    5, 8, 3, 1, 0, 7, 3, 4, 4, 3, 7, 0, 6, 0, 1, 2, 2, 1, 0, 6};
constexpr uint8_t kFlipDiag[] = {
    0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};
constexpr uint8_t kLower[] = {
    28, 0,  1,  2,  3,  4,  5,  6,  0, 29, 7,  8,  9,  10, 11, 12,
    1,  7,  30, 13, 14, 15, 16, 17, 2, 8,  13, 31, 18, 19, 20, 21,
    3,  9,  14, 18, 32, 22, 23, 24, 4, 10, 15, 19, 22, 33, 25, 26,
    5,  11, 16, 20, 23, 25, 34, 27, 6, 12, 17, 21, 24, 26, 27, 35};
constexpr uint8_t kDiag[] = {
    0, 0, 0, 0, 0, 0,  0,  8, 0, 1, 0, 0, 0,  0,  9, 0, 0, 0, 2, 0, 0,  10,
    0, 0, 0, 0, 0, 3,  11, 0, 0, 0, 0, 0, 0,  12, 4, 0, 0, 0, 0, 0, 13, 0,
    0, 5, 0, 0, 0, 14, 0,  0, 0, 0, 6, 0, 15, 0,  0, 0, 0, 0, 0, 7};
constexpr uint8_t kFlap[2][64] = {
    {0, 0,  0,  0,  0,  0,  0,  0, 0, 6,  12, 18, 18, 12, 6,  0,
     1, 7,  13, 19, 19, 13, 7,  1, 2, 8,  14, 20, 20, 14, 8,  2,
     3, 9,  15, 21, 21, 15, 9,  3, 4, 10, 16, 22, 22, 16, 10, 4,
     5, 11, 17, 23, 23, 17, 11, 5, 0, 0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  2,  1,  0,
     4,  5,  6,  7,  7,  6,  5,  4,  8,  9,  10, 11, 11, 10, 9,  8,
     12, 13, 14, 15, 15, 14, 13, 12, 16, 17, 18, 19, 19, 18, 17, 16,
     20, 21, 22, 23, 23, 22, 21, 20, 0,  0,  0,  0,  0,  0,  0,  0}};
constexpr uint8_t kPawnTwist[2][64] = {
    {0,  0,  0,  0, 0, 0,  0,  0,  47, 35, 23, 11, 10, 22, 34, 46,
     45, 33, 21, 9, 8, 20, 32, 44, 43, 31, 19, 7,  6,  18, 30, 42,
     41, 29, 17, 5, 4, 16, 28, 40, 39, 27, 15, 3,  2,  14, 26, 38,
     37, 25, 13, 1, 0, 12, 24, 36, 0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  47, 45, 43, 41, 40, 42, 44, 46,
     39, 37, 35, 33, 32, 34, 36, 38, 31, 29, 27, 25, 24, 26, 28, 30,
     23, 21, 19, 17, 16, 18, 20, 22, 15, 13, 11, 9,  8,  10, 12, 14,
     7,  5,  3,  1,  0,  2,  4,  6,  0,  0,  0,  0,  0,  0,  0,  0}};
constexpr int16_t kKKIdx[10][64] = {
    {-1, -1, -1, 0,  1,  2,  3,  4,  -1, -1, -1, 5,  6,  7,  8,  9,
     10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57},
    {58,  -1,  -1,  -1,  59,  60,  61,  62,  63,  -1,  -1,  -1,  64,
     65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
     78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
     91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103,
     104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115},
    {116, 117, -1,  -1,  -1,  118, 119, 120, 121, 122, -1,  -1,  -1,
     123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
     136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
     149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
     162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173},
    {174, -1,  -1,  -1,  175, 176, 177, 178, 179, -1,  -1,  -1,  180,
     181, 182, 183, 184, -1,  -1,  -1,  185, 186, 187, 188, 189, 190,
     191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
     204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
     217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228},
    {229, 230, -1,  -1,  -1,  231, 232, 233, 234, 235, -1,  -1,  -1,
     236, 237, 238, 239, 240, -1,  -1,  -1,  241, 242, 243, 244, 245,
     246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
     259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
     272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283},
    {284, 285, 286, 287, 288, 289, 290, 291, 292, 293, -1,  -1,  -1,
     294, 295, 296, 297, 298, -1,  -1,  -1,  299, 300, 301, 302, 303,
     -1,  -1,  -1,  304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
     314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
     327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338},
    {-1,  -1,  339, 340, 341, 342, 343, 344, -1,  -1,  345, 346, 347,
     348, 349, 350, -1,  -1,  441, 351, 352, 353, 354, 355, -1,  -1,
     -1,  442, 356, 357, 358, 359, -1,  -1,  -1,  -1,  443, 360, 361,
     362, -1,  -1,  -1,  -1,  -1,  444, 363, 364, -1,  -1,  -1,  -1,
     -1,  -1,  445, 365, -1,  -1,  -1,  -1,  -1,  -1,  -1,  446},
    {-1, -1, -1, 366, 367, 368, 369, 370, -1, -1, -1, 371, 372, 373, 374, 375,
     -1, -1, -1, 376, 377, 378, 379, 380, -1, -1, -1, 447, 381, 382, 383, 384,
     -1, -1, -1, -1,  448, 385, 386, 387, -1, -1, -1, -1,  -1,  449, 388, 389,
     -1, -1, -1, -1,  -1,  -1,  450, 390, -1, -1, -1, -1,  -1,  -1,  -1,  451},
    {452, 391, 392, 393, 394, 395, 396, 397, -1,  -1,  -1,  -1,  398,
     399, 400, 401, -1,  -1,  -1,  -1,  402, 403, 404, 405, -1,  -1,
     -1,  -1,  406, 407, 408, 409, -1,  -1,  -1,  -1,  453, 410, 411,
     412, -1,  -1,  -1,  -1,  -1,  454, 413, 414, -1,  -1,  -1,  -1,
     -1,  -1,  455, 415, -1,  -1,  -1,  -1,  -1,  -1,  -1,  456},
    {457, 416, 417, 418, 419, 420, 421, 422, -1,  458, 423, 424, 425,
     426, 427, 428, -1,  -1,  -1,  -1,  -1,  429, 430, 431, -1,  -1,
     -1,  -1,  -1,  432, 433, 434, -1,  -1,  -1,  -1,  -1,  435, 436,
     437, -1,  -1,  -1,  -1,  -1,  459, 438, 439, -1,  -1,  -1,  -1,
     -1,  -1,  460, 440, -1,  -1,  -1,  -1,  -1,  -1,  -1,  461}};
constexpr uint8_t kFileToFile[] = {0, 1, 2, 3, 3, 2, 1, 0};
constexpr int kWdlToMap[5] = {1, 3, 0, 2, 0};
constexpr uint8_t kPAFlags[5] = {8, 0, 0, 0, 4};
size_t Binomial[7][64];
size_t PawnIdx[2][6][24];
size_t PawnFactorFile[6][4];
size_t PawnFactorRank[6][6];
Key MaterialHash[16][64];
void init_indices() {
  // Binomial[k][n] = Bin(n, k)
  for (int i = 0; i < 7; i++)
    for (int j = 0; j < 64; j++) {
      size_t f = 1;
      size_t l = 1;
      for (int k = 0; k < i; k++) {
        f *= (j - k);
        l *= (k + 1);
      }
      Binomial[i][j] = f / l;
    }
  for (int i = 0; i < 6; i++) {
    size_t s = 0;
    for (int j = 0; j < 24; j++) {
      PawnIdx[0][i][j] = s;
      s += Binomial[i][kPawnTwist[0][(1 + (j % 6)) * 8 + (j / 6)]];
      if ((j + 1) % 6 == 0) {
        PawnFactorFile[i][j / 6] = s;
        s = 0;
      }
    }
  }
  for (int i = 0; i < 6; i++) {
    size_t s = 0;
    for (int j = 0; j < 24; j++) {
      PawnIdx[1][i][j] = s;
      s += Binomial[i][kPawnTwist[1][(1 + (j / 4)) * 8 + (j % 4)]];
      if ((j + 1) % 4 == 0) {
        PawnFactorRank[i][j / 4] = s;
        s = 0;
      }
    }
  }
  // TODO: choose a good seed.
  std::mt19937 gen(123523465);
  std::uniform_int_distribution<Key> dist(std::numeric_limits<Key>::lowest(),
                                          std::numeric_limits<Key>::max());
  for (int i = 0; i < 16; i++) {
    // MaterialHash for 0 instances of a piece is 0 as an optimization so
    // calc_key_from_pieces doesn't have to add in all the missing pieces.
    MaterialHash[i][0] = 0;
    for (int j = 1; j < 64; j++) {
      MaterialHash[i][j] = dist(gen);
    }
  }
}
std::once_flag indicies_flag;
void initonce_indicies() { std::call_once(indicies_flag, init_indices); }
// Produce a 64-bit material key corresponding to the material combination
// defined by pcs[16], where pcs[1], ..., pcs[6] are the number of white
// pawns, ..., kings and pcs[9], ..., pcs[14] are the number of black
// pawns, ..., kings.
Key calc_key_from_pcs(int* pcs, bool flip) {
  Key key = 0;
  const int color = !flip ? 0 : 8;
  for (int i = W_PAWN; i <= B_KING; i++) key += MaterialHash[i][pcs[i ^ color]];
  return key;
}
// Produce a 64-bit material key corresponding to the material combination
// piece[0], ..., piece[num - 1], where each value corresponds to a piece
// (1-6 for white pawn-king, 9-14 for black pawn-king).
Key calc_key_from_pieces(uint8_t* piece, int num) {
  Key key = 0;
  for (int i = 0; i < num; i++) {
    if (piece[i]) key += MaterialHash[piece[i]][1];
  }
  return key;
}
Key calc_key_from_position(const ChessBoard& pos) {
  Key key = 0;
  const bool flipped = pos.flipped();
  for (int i = PAWN; i <= KING; i++) {
    // White pieces - ours if not flipped.
    key += MaterialHash[i][count_pieces(pos, i, flipped)];
    // Black pieces - ours if flipped.
    key += MaterialHash[i + 8][count_pieces(pos, i, !flipped)];
  }
  return key;
}
int leading_pawn(int* p, BaseEntry* be, const int enc) {
  for (int i = 1; i < be->pawns[0]; i++) {
    if (kFlap[enc - 1][p[0]] > kFlap[enc - 1][p[i]]) Swap(p[0], p[i]);
  }
  return enc == FILE_ENC ? kFileToFile[p[0] & 7] : (p[0] - 8) >> 3;
}
size_t encode(int* p, EncInfo* ei, BaseEntry* be, const int enc) {
  const int n = be->num;
  size_t idx;
  int k;
  if (p[0] & 0x04) {
    for (int i = 0; i < n; i++) p[i] ^= 0x07;
  }
  if (enc == PIECE_ENC) {
    if (p[0] & 0x20) {
      for (int i = 0; i < n; i++) p[i] ^= 0x38;
    }
    for (int i = 0; i < n; i++) {
      if (kOffDiag[p[i]]) {
        if (kOffDiag[p[i]] > 0 && i < (be->kk_enc ? 2 : 3)) {
          for (int j = 0; j < n; j++) p[j] = kFlipDiag[p[j]];
        }
        break;
      }
    }
    if (be->kk_enc) {
      idx = kKKIdx[kTriangle[p[0]]][p[1]];
      k = 2;
    } else {
      const int s1 = (p[1] > p[0]);
      const int s2 = (p[2] > p[0]) + (p[2] > p[1]);
      if (kOffDiag[p[0]]) {
        idx = kTriangle[p[0]] * 63 * 62 + (p[1] - s1) * 62 + (p[2] - s2);
      } else if (kOffDiag[p[1]]) {
        idx =
            6 * 63 * 62 + kDiag[p[0]] * 28 * 62 + kLower[p[1]] * 62 + p[2] - s2;
      } else if (kOffDiag[p[2]]) {
        idx = 6 * 63 * 62 + 4 * 28 * 62 + kDiag[p[0]] * 7 * 28 +
              (kDiag[p[1]] - s1) * 28 + kLower[p[2]];
      } else {
        idx = 6 * 63 * 62 + 4 * 28 * 62 + 4 * 7 * 28 + kDiag[p[0]] * 7 * 6 +
              (kDiag[p[1]] - s1) * 6 + (kDiag[p[2]] - s2);
      }
      k = 3;
    }
    idx *= ei->factor[0];
  } else {
    for (int i = 1; i < be->pawns[0]; i++) {
      for (int j = i + 1; j < be->pawns[0]; j++) {
        if (kPawnTwist[enc - 1][p[i]] < kPawnTwist[enc - 1][p[j]]) {
          Swap(p[i], p[j]);
        }
      }
    }
    k = be->pawns[0];
    idx = PawnIdx[enc - 1][k - 1][kFlap[enc - 1][p[0]]];
    for (int i = 1; i < k; i++) {
      idx += Binomial[k - i][kPawnTwist[enc - 1][p[i]]];
    }
    idx *= ei->factor[0];
    // Pawns of other color
    if (be->pawns[1]) {
      const int t = k + be->pawns[1];
      for (int i = k; i < t; i++) {
        for (int j = i + 1; j < t; j++) {
          if (p[i] > p[j]) Swap(p[i], p[j]);
        }
      }
      size_t s = 0;
      for (int i = k; i < t; i++) {
        const int sq = p[i];
        int skips = 0;
        for (int j = 0; j < k; j++) skips += (sq > p[j]);
        s += Binomial[i - k + 1][sq - skips - 8];
      }
      idx += s * ei->factor[k];
      k = t;
    }
  }
  for (; k < n;) {
    const int t = k + ei->norm[k];
    for (int i = k; i < t; i++) {
      for (int j = i + 1; j < t; j++) {
        if (p[i] > p[j]) Swap(p[i], p[j]);
      }
    }
    size_t s = 0;
    for (int i = k; i < t; i++) {
      const int sq = p[i];
      int skips = 0;
      for (int j = 0; j < k; j++) skips += (sq > p[j]);
      s += Binomial[i - k + 1][sq - skips];
    }
    idx += s * ei->factor[k];
    k = t;
  }
  return idx;
}
size_t encode_piece(int* p, EncInfo* ei, BaseEntry* be) {
  return encode(p, ei, be, PIECE_ENC);
}
size_t encode_pawn_f(int* p, EncInfo* ei, BaseEntry* be) {
  return encode(p, ei, be, FILE_ENC);
}
size_t encode_pawn_r(int* p, EncInfo* ei, BaseEntry* be) {
  return encode(p, ei, be, RANK_ENC);
}
// Count number of placements of k like pieces on n squares
size_t subfactor(size_t k, size_t n) {
  size_t f = n;
  size_t l = 1;
  for (size_t i = 1; i < k; i++) {
    f *= n - i;
    l *= i + 1;
  }
  return f / l;
}
size_t init_enc_info(EncInfo* ei, BaseEntry* be, uint8_t* tb, int shift, int t,
                     const int enc) {
  const bool more_pawns = enc != PIECE_ENC && be->pawns[1] > 0;
  for (int i = 0; i < be->num; i++) {
    ei->pieces[i] = (tb[i + 1 + more_pawns] >> shift) & 0x0f;
    ei->norm[i] = 0;
  }
  const int order = (tb[0] >> shift) & 0x0f;
  const int order2 = more_pawns ? (tb[1] >> shift) & 0x0f : 0x0f;
  int k = ei->norm[0] = enc != PIECE_ENC ? be->pawns[0] : be->kk_enc ? 2 : 3;
  if (more_pawns) {
    ei->norm[k] = be->pawns[1];
    k += ei->norm[k];
  }
  for (int i = k; i < be->num; i += ei->norm[i]) {
    for (int j = i; j < be->num && ei->pieces[j] == ei->pieces[i]; j++) {
      ei->norm[i]++;
    }
  }
  int n = 64 - k;
  size_t f = 1;
  for (int i = 0; k < be->num || i == order || i == order2; i++) {
    if (i == order) {
      ei->factor[0] = f;
      f *= enc == FILE_ENC
               ? PawnFactorFile[ei->norm[0] - 1][t]
               : enc == RANK_ENC ? PawnFactorRank[ei->norm[0] - 1][t]
                                 : be->kk_enc ? 462 : 31332;
    } else if (i == order2) {
      ei->factor[ei->norm[0]] = f;
      f *= subfactor(ei->norm[ei->norm[0]], 48 - ei->norm[0]);
    } else {
      ei->factor[k] = f;
      f *= subfactor(ei->norm[k], n);
      n -= ei->norm[k];
      k += ei->norm[k];
    }
  }
  return f;
}
void calc_symLen(PairsData* d, uint32_t s, char* tmp) {
  uint8_t* w = d->symPat + 3 * s;
  const uint32_t s2 = (w[2] << 4) | (w[1] >> 4);
  if (s2 == 0x0fff)
    d->symLen[s] = 0;
  else {
    const uint32_t s1 = ((w[1] & 0xf) << 8) | w[0];
    if (!tmp[s1]) calc_symLen(d, s1, tmp);
    if (!tmp[s2]) calc_symLen(d, s2, tmp);
    d->symLen[s] = d->symLen[s1] + d->symLen[s2] + 1;
  }
  tmp[s] = 1;
}
int is_little_endian() {
  union {
    uint32_t i;
    uint8_t byte[4];
  } num_union = {0x01020304};
  return num_union.byte[0] == 4;
}
template <typename T, int Half = sizeof(T) / 2, int End = sizeof(T) - 1>
T swap_endian(T val) {
  static_assert(std::is_unsigned<T>::value,
                "Argument of swap_endian not unsigned");
  T x = val;
  uint8_t tmp, *c = (uint8_t*)&x;
  for (int i = 0; i < Half; ++i) {
    tmp = c[i], c[i] = c[End - i], c[End - i] = tmp;
  }
  return x;
}
uint32_t from_le_u32(uint32_t v) {
  return is_little_endian() ? v : swap_endian(v);
}
uint16_t from_le_u16(uint16_t v) {
  return is_little_endian() ? v : swap_endian(v);
}
uint64_t from_be_u64(uint64_t v) {
  return is_little_endian() ? swap_endian(v) : v;
}
uint32_t from_be_u32(uint32_t v) {
  return is_little_endian() ? swap_endian(v) : v;
}
uint32_t read_le_u32(void* p) {
  return from_le_u32(*static_cast<uint32_t*>(p));
}
uint16_t read_le_u16(void* p) {
  return from_le_u16(*static_cast<uint16_t*>(p));
}
PairsData* setup_pairs(uint8_t** ptr, size_t tb_size, size_t* size,
                       uint8_t* flags, int type) {
  PairsData* d;
  uint8_t* data = *ptr;
  *flags = data[0];
  if (data[0] & 0x80) {
    d = static_cast<PairsData*>(malloc(sizeof(*d)));
    d->idxBits = 0;
    d->constValue[0] = type == WDL ? data[1] : 0;
    d->constValue[1] = 0;
    *ptr = data + 2;
    size[0] = size[1] = size[2] = 0;
    return d;
  }
  const uint8_t block_size = data[1];
  const uint8_t idx_bits = data[2];
  const uint32_t real_num_blocks = read_le_u32(&data[4]);
  const uint32_t num_blocks = real_num_blocks + data[3];
  const int max_len = data[8];
  const int min_len = data[9];
  const int h = max_len - min_len + 1;
  const uint32_t num_syms = read_le_u16(&data[10 + 2 * h]);
  d = static_cast<PairsData*>(
      malloc(sizeof(*d) + h * sizeof(uint64_t) + num_syms));
  d->blockSize = block_size;
  d->idxBits = idx_bits;
  d->offset = reinterpret_cast<uint16_t*>(&data[10]);
  d->symLen = reinterpret_cast<uint8_t*>(d) + sizeof(*d) + h * sizeof(uint64_t);
  d->symPat = &data[12 + 2 * h];
  d->minLen = min_len;
  *ptr = &data[12 + 2 * h + 3 * num_syms + (num_syms & 1)];
  const size_t num_indices = (tb_size + (1ULL << idx_bits) - 1) >> idx_bits;
  size[0] = 6ULL * num_indices;
  size[1] = 2ULL * num_blocks;
  size[2] = static_cast<size_t>(real_num_blocks) << block_size;
  std::vector<char> tmp;
  tmp.resize(num_syms);
  memset(tmp.data(), 0, num_syms);
  for (uint32_t s = 0; s < num_syms; s++) {
    if (!tmp[s]) calc_symLen(d, s, tmp.data());
  }
  d->base[h - 1] = 0;
  for (int i = h - 2; i >= 0; i--) {
    d->base[i] = (d->base[i + 1] +
                  read_le_u16(reinterpret_cast<uint8_t*>(d->offset + i)) -
                  read_le_u16(reinterpret_cast<uint8_t*>(d->offset + i + 1))) /
                 2;
  }
  for (int i = 0; i < h; i++) d->base[i] <<= 64 - (min_len + i);
  d->offset -= d->minLen;
  return d;
}
uint8_t* decompress_pairs(PairsData* d, size_t idx) {
  if (!d->idxBits) return d->constValue;
  const uint32_t main_idx = idx >> d->idxBits;
  int lit_idx = (idx & ((static_cast<size_t>(1) << d->idxBits) - 1)) -
                (static_cast<size_t>(1) << (d->idxBits - 1));
  uint32_t block;
  memcpy(&block, d->indexTable + 6 * main_idx, sizeof(block));
  block = from_le_u32(block);
  const uint16_t idx_offset =
      *reinterpret_cast<uint16_t*>(d->indexTable + 6 * main_idx + 4);
  lit_idx += from_le_u16(idx_offset);
  if (lit_idx < 0) {
    while (lit_idx < 0) lit_idx += d->sizeTable[--block] + 1;
  } else {
    while (lit_idx > d->sizeTable[block]) lit_idx -= d->sizeTable[block++] + 1;
  }
  uint32_t* ptr = reinterpret_cast<uint32_t*>(
      d->data + (static_cast<size_t>(block) << d->blockSize));
  const int m = d->minLen;
  uint16_t* offset = d->offset;
  uint64_t* base = d->base - m;
  uint8_t* sym_len = d->symLen;
  uint32_t sym;
  uint32_t bit_cnt = 0;  // number of "empty bits" in code
  uint64_t code = from_be_u64(*reinterpret_cast<uint64_t*>(ptr));
  ptr += 2;
  for (;;) {
    int l = m;
    while (code < base[l]) l++;
    sym = from_le_u16(offset[l]);
    sym += (code - base[l]) >> (64 - l);
    if (lit_idx < static_cast<int>(sym_len[sym]) + 1) break;
    lit_idx -= static_cast<int>(sym_len[sym]) + 1;
    code <<= l;
    bit_cnt += l;
    if (bit_cnt >= 32) {
      bit_cnt -= 32;
      const uint32_t tmp = from_be_u32(*ptr++);
      code |= static_cast<uint64_t>(tmp) << bit_cnt;
    }
  }
  uint8_t* symPat = d->symPat;
  while (sym_len[sym] != 0) {
    uint8_t* w = symPat + (3 * sym);
    const int s1 = ((w[1] & 0xf) << 8) | w[0];
    if (lit_idx < static_cast<int>(sym_len[s1]) + 1) {
      sym = s1;
    } else {
      lit_idx -= static_cast<int>(sym_len[s1]) + 1;
      sym = (w[2] << 4) | (w[1] >> 4);
    }
  }
  return &symPat[3 * sym];
}
// p[i] is to contain the square 0-63 (A1-H8) for a piece of type
// pc[i] ^ flip, where 1 = white pawn, ..., 14 = black king and pc ^ flip
// flips between white and black if flip == true.
// Pieces of the same type are guaranteed to be consecutive.
int fill_squares(const ChessBoard& pos, uint8_t* pc, bool flip, int mirror,
                 int* p, int i) {
  // if pos.flipped the board is already mirrored, so mirror it again.
  if (pos.flipped()) mirror ^= 0x38;
  BitBoard bb = pieces(pos, pc[i] & 7,
                       static_cast<bool>((pc[i] >> 3)) ^ flip ^ pos.flipped());
  for (auto sq : bb) {
    p[i++] = sq.as_int() ^ mirror;
  }
  return i;
}
}  // namespace
class SyzygyTablebaseImpl {
 public:
  SyzygyTablebaseImpl(const std::string& paths)
      : piece_entries_(TB_MAX_PIECE), pawn_entries_(TB_MAX_PAWN) {
    initonce_indicies();
    if (paths.size() == 0 || paths == "<empty>") return;
    paths_ = paths;
    tb_hash_.resize(1 << TB_HASHBITS);
    char str[33];
    for (int i = 0; i < 5; i++) {
      sprintf(str, "K%cvK", pchr(i));
      init_tb(str);
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        sprintf(str, "K%cvK%c", pchr(i), pchr(j));
        init_tb(str);
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        sprintf(str, "K%c%cvK", pchr(i), pchr(j));
        init_tb(str);
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = 0; k < 5; k++) {
          sprintf(str, "K%c%cvK%c", pchr(i), pchr(j), pchr(k));
          init_tb(str);
        }
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          sprintf(str, "K%c%c%cvK", pchr(i), pchr(j), pchr(k));
          init_tb(str);
        }
      }
    }
    // 6- and 7-piece TBs make sense only with a 64-bit address space
    if (sizeof(size_t) < 8 || TB_PIECES < 6) goto finished;
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = i; k < 5; k++) {
          for (int l = (i == k) ? j : k; l < 5; l++) {
            sprintf(str, "K%c%cvK%c%c", pchr(i), pchr(j), pchr(k), pchr(l));
            init_tb(str);
          }
        }
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = 0; l < 5; l++) {
            sprintf(str, "K%c%c%cvK%c", pchr(i), pchr(j), pchr(k), pchr(l));
            init_tb(str);
          }
        }
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = k; l < 5; l++) {
            sprintf(str, "K%c%c%c%cvK", pchr(i), pchr(j), pchr(k), pchr(l));
            init_tb(str);
          }
        }
      }
    }
    if (TB_PIECES < 7) goto finished;
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = k; l < 5; l++) {
            for (int m = l; m < 5; m++) {
              sprintf(str, "K%c%c%c%c%cvK", pchr(i), pchr(j), pchr(k), pchr(l),
                      pchr(m));
              init_tb(str);
            }
          }
        }
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = k; l < 5; l++) {
            for (int m = 0; m < 5; m++) {
              sprintf(str, "K%c%c%c%cvK%c", pchr(i), pchr(j), pchr(k), pchr(l),
                      pchr(m));
              init_tb(str);
            }
          }
        }
      }
    }
    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = 0; l < 5; l++) {
            for (int m = l; m < 5; m++) {
              sprintf(str, "K%c%c%cvK%c%c", pchr(i), pchr(j), pchr(k), pchr(l),
                      pchr(m));
              init_tb(str);
            }
          }
        }
      }
    }
  finished:
    CERR << "Found " << num_wdl_ << " WDL, " << num_dtm_ << " DTM and "
         << num_dtz_ << " DTZ tablebase files.";
  }
  ~SyzygyTablebaseImpl() {
    // if pathString was set there may be entries in need of cleaning.
    if (!paths_.empty()) {
      for (int i = 0; i < num_piece_entries_; i++)
        free_tb_entry(&piece_entries_[i]);
      for (int i = 0; i < num_pawn_entries_; i++)
        free_tb_entry(&pawn_entries_[i]);
    }
  }
  int max_cardinality() const { return max_cardinality_; }
  int probe_wdl_table(const ChessBoard& pos, int* success) {
    return probe_table(pos, 0, success, WDL);
  }
  int probe_dtm_table(const ChessBoard& pos, int won, int* success) {
    return probe_table(pos, won, success, DTM);
  }
  int probe_dtz_table(const ChessBoard& pos, int wdl, int* success) {
    return probe_table(pos, wdl, success, DTZ);
  }
 private:
  std::string name_for_tb(const char* str, const char* suffix) {
    std::stringstream path_string_stream(paths_);
    std::string path;
    std::ifstream stream;
    while (std::getline(path_string_stream, path, SEP_CHAR)) {
      std::string fname = path + "/" + str + suffix;
      stream.open(fname);
      if (stream.is_open()) return fname;
    }
    return std::string();
  }
  bool test_tb(const char* str, const char* suffix) {
    return !name_for_tb(str, suffix).empty();
  }
  void* map_tb(const char* name, const char* suffix, map_t* mapping) {
    std::string fname = name_for_tb(name, suffix);
    void* base_address;
#ifndef _WIN32
    struct stat statbuf;
    int fd = ::open(fname.c_str(), O_RDONLY);
    if (fd == -1) return nullptr;
    fstat(fd, &statbuf);
    if (statbuf.st_size % 64 != 16) {
      throw Exception("Corrupt tablebase file " + fname);
    }
    *mapping = statbuf.st_size;
    base_address = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base_address == MAP_FAILED) {
      throw Exception("Could not mmap() " + fname);
    }
#else
    const HANDLE fd =
        CreateFileA(fname.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fd == INVALID_HANDLE_VALUE) return nullptr;
    DWORD size_high;
    DWORD size_low = GetFileSize(fd, &size_high);
    if (size_low % 64 != 16) {
      throw Exception("Corrupt tablebase file " + fname);
    }
    HANDLE mmap = CreateFileMapping(fd, nullptr, PAGE_READONLY, size_high,
                                    size_low, nullptr);
    CloseHandle(fd);
    if (!mmap) {
      throw Exception("CreateFileMapping() failed");
    }
    *mapping = mmap;
    base_address = MapViewOfFile(mmap, FILE_MAP_READ, 0, 0, 0);
    if (!base_address) {
      throw Exception("MapViewOfFile() failed, name = " + fname + ", error = " + std::to_string(GetLastError()));
    }
#endif
    return base_address;
  }
  void unmap_file(void* base_address, map_t mapping) {
#ifndef _WIN32
    munmap(base_address, mapping);
#else
    UnmapViewOfFile(base_address);
    CloseHandle(mapping);
#endif
  }
  void add_to_hash(BaseEntry* ptr, Key key) {
    int idx;
    idx = key >> (64 - TB_HASHBITS);
    while (tb_hash_[idx].ptr) idx = (idx + 1) & ((1 << TB_HASHBITS) - 1);
    tb_hash_[idx].key = key;
    tb_hash_[idx].ptr = ptr;
  }
  void init_tb(char* str) {
    if (!test_tb(str, kSuffix[WDL])) return;
    int pcs[16];
    for (int i = 0; i < 16; i++) pcs[i] = 0;
    int color = 0;
    for (char* s = str; *s; s++) {
      if (*s == 'v') {
        color = 8;
      } else {
        for (int i = PAWN; i <= KING; i++) {
          if (*s == kPieceToChar[i]) {
            pcs[i | color]++;
            break;
          }
        }
      }
    }
    const Key key = calc_key_from_pcs(pcs, false);
    const Key key2 = calc_key_from_pcs(pcs, true);
    const bool has_pawns = pcs[W_PAWN] || pcs[B_PAWN];
    BaseEntry* be =
        has_pawns
            ? static_cast<BaseEntry*>(&pawn_entries_[num_pawn_entries_++])
            : static_cast<BaseEntry*>(&piece_entries_[num_piece_entries_++]);
    be->hasPawns = has_pawns;
    be->key = key;
    be->symmetric = key == key2;
    be->num = 0;
    for (int i = 0; i < 16; i++) be->num += pcs[i];
    num_wdl_++;
    num_dtm_ += be->hasDtm = test_tb(str, kSuffix[DTM]);
    num_dtz_ += be->hasDtz = test_tb(str, kSuffix[DTZ]);
    max_cardinality_ = std::max(max_cardinality_, static_cast<int>(be->num));
    if (be->hasDtm)
      max_cardinality_dtm_ =
          std::max(max_cardinality_dtm_, static_cast<int>(be->num));
    for (int type = 0; type < 3; type++) be->ready[type] = 0;
    if (!be->hasPawns) {
      int j = 0;
      for (int i = 0; i < 16; i++) {
        if (pcs[i] == 1) j++;
      }
      be->kk_enc = j == 2;
    } else {
      be->pawns[0] = pcs[W_PAWN];
      be->pawns[1] = pcs[B_PAWN];
      if (pcs[B_PAWN] && (!pcs[W_PAWN] || pcs[W_PAWN] > pcs[B_PAWN])) {
        Swap(be->pawns[0], be->pawns[1]);
      }
    }
    add_to_hash(be, key);
    if (key != key2) add_to_hash(be, key2);
  }
  void free_tb_entry(BaseEntry* be) {
    for (int type = 0; type < 3; type++) {
      if (atomic_load_explicit(&be->ready[type], std::memory_order_relaxed)) {
        unmap_file(be->data[type], be->mapping[type]);
        const int num = num_tables(be, type);
        EncInfo* ei = first_ei(be, type);
        for (int t = 0; t < num; t++) {
          free(ei[t].precomp);
          if (type != DTZ) free(ei[num + t].precomp);
        }
        atomic_store_explicit(&be->ready[type], false,
                              std::memory_order_relaxed);
      }
    }
  }
  bool init_table(BaseEntry* be, const char* str, int type) {
    uint8_t* data =
        static_cast<uint8_t*>(map_tb(str, kSuffix[type], &be->mapping[type]));
    if (!data) return false;
    if (read_le_u32(data) != kMagic[type]) {
      fprintf(stderr, "Corrupted table.\n");
      unmap_file(data, be->mapping[type]);
      return false;
    }
    be->data[type] = data;
    const bool split = type != DTZ && (data[4] & 0x01);
    if (type == DTM) be->dtmLossOnly = data[4] & 0x04;
    data += 5;
    size_t tb_size[6][2];
    const int num = num_tables(be, type);
    EncInfo* ei = first_ei(be, type);
    const int enc = !be->hasPawns ? PIECE_ENC : type != DTM ? FILE_ENC : RANK_ENC;
    for (int t = 0; t < num; t++) {
      tb_size[t][0] = init_enc_info(&ei[t], be, data, 0, t, enc);
      if (split) {
        tb_size[t][1] = init_enc_info(&ei[num + t], be, data, 4, t, enc);
      }
      data += be->num + 1 + (be->hasPawns && be->pawns[1]);
    }
    data += (uintptr_t)data & 1;
    size_t size[6][2][3];
    for (int t = 0; t < num; t++) {
      uint8_t flags;
      ei[t].precomp =
          setup_pairs(&data, tb_size[t][0], size[t][0], &flags, type);
      if (type == DTZ) {
        if (!be->hasPawns) {
          PIECE(be)->dtzFlags = flags;
        } else {
          PAWN(be)->dtzFlags[t] = flags;
        }
      }
      if (split) {
        ei[num + t].precomp =
            setup_pairs(&data, tb_size[t][1], size[t][1], &flags, type);
      } else if (type != DTZ) {
        ei[num + t].precomp = NULL;
      }
    }
    if (type == DTM && !be->dtmLossOnly) {
      uint16_t* map = reinterpret_cast<uint16_t*>(data);
      *(be->hasPawns ? &PAWN(be)->dtmMap : &PIECE(be)->dtmMap) = map;
      uint16_t(*mapIdx)[2][2] =
          be->hasPawns ? &PAWN(be)->dtmMapIdx[0] : &PIECE(be)->dtmMapIdx;
      for (int t = 0; t < num; t++) {
        for (int i = 0; i < 2; i++) {
          mapIdx[t][0][i] = reinterpret_cast<uint16_t*>(data) + 1 - map;
          data += 2 + 2 * read_le_u16(data);
        }
        if (split) {
          for (int i = 0; i < 2; i++) {
            mapIdx[t][1][i] = reinterpret_cast<uint16_t*>(data) + 1 - map;
            data += 2 + 2 * read_le_u16(data);
          }
        }
      }
    }
    if (type == DTZ) {
      void* map = data;
      *(be->hasPawns ? &PAWN(be)->dtzMap : &PIECE(be)->dtzMap) = map;
      uint16_t(*mapIdx)[4] =
          be->hasPawns ? &PAWN(be)->dtzMapIdx[0] : &PIECE(be)->dtzMapIdx;
      uint8_t* flags =
          be->hasPawns ? &PAWN(be)->dtzFlags[0] : &PIECE(be)->dtzFlags;
      for (int t = 0; t < num; t++) {
        if (flags[t] & 2) {
          if (!(flags[t] & 16)) {
            for (int i = 0; i < 4; i++) {
              mapIdx[t][i] = data + 1 - static_cast<uint8_t*>(map);
              data += 1 + data[0];
            }
          } else {
            data += reinterpret_cast<uintptr_t>(data) & 0x01;
            for (int i = 0; i < 4; i++) {
              mapIdx[t][i] = reinterpret_cast<uint16_t*>(data) + 1 -
                             static_cast<uint16_t*>(map);
              data += 2 + 2 * read_le_u16(data);
            }
          }
        }
      }
      data += reinterpret_cast<uintptr_t>(data) & 0x01;
    }
    for (int t = 0; t < num; t++) {
      ei[t].precomp->indexTable = data;
      data += size[t][0][0];
      if (split) {
        ei[num + t].precomp->indexTable = data;
        data += size[t][1][0];
      }
    }
    for (int t = 0; t < num; t++) {
      ei[t].precomp->sizeTable = reinterpret_cast<uint16_t*>(data);
      data += size[t][0][1];
      if (split) {
        ei[num + t].precomp->sizeTable = reinterpret_cast<uint16_t*>(data);
        data += size[t][1][1];
      }
    }
    for (int t = 0; t < num; t++) {
      data = reinterpret_cast<uint8_t*>(
          (reinterpret_cast<uintptr_t>(data) + 0x3f) & ~0x3f);
      ei[t].precomp->data = data;
      data += size[t][0][2];
      if (split) {
        data = reinterpret_cast<uint8_t*>(
            (reinterpret_cast<uintptr_t>(data) + 0x3f) & ~0x3f);
        ei[num + t].precomp->data = data;
        data += size[t][1][2];
      }
    }
    if (type == DTM && be->hasPawns) {
      PAWN(be)->dtmSwitched =
          calc_key_from_pieces(ei[0].pieces, be->num) != be->key;
    }
    return true;
  }
  int probe_table(const ChessBoard& pos, int s, int* success, const int type) {
    // Obtain the position's material-signature key
    const Key key = calc_key_from_position(pos);
    // Test for KvK
    if (type == WDL && (pos.ours() | pos.theirs()) == pos.kings()) {
      return 0;
    }
    int hash_idx = key >> (64 - TB_HASHBITS);
    while (tb_hash_[hash_idx].key && tb_hash_[hash_idx].key != key) {
      hash_idx = (hash_idx + 1) & ((1 << TB_HASHBITS) - 1);
    }
    if (!tb_hash_[hash_idx].ptr) {
      *success = 0;
      return 0;
    }
    BaseEntry* be = tb_hash_[hash_idx].ptr;
    if ((type == DTM && !be->hasDtm) || (type == DTZ && !be->hasDtz)) {
      *success = 0;
      return 0;
    }
    // Use double-checked locking to reduce locking overhead
    if (!atomic_load_explicit(&be->ready[type], std::memory_order_acquire)) {
      Mutex::Lock lock(ready_mutex_);
      if (!atomic_load_explicit(&be->ready[type], std::memory_order_relaxed)) {
        char str[16];
        prt_str(pos, str, be->key != key);
        if (!init_table(be, str, type)) {
          tb_hash_[hash_idx].ptr = nullptr;  // mark as deleted
          *success = 0;
          return 0;
        }
        atomic_store_explicit(&be->ready[type], true,
                              std::memory_order_release);
      }
    }
    bool bside, flip;
    if (!be->symmetric) {
      flip = key != be->key;
      bside = (!pos.flipped()) == flip;
      if (type == DTM && be->hasPawns && PAWN(be)->dtmSwitched) {
        flip = !flip;
        bside = !bside;
      }
    } else {
      flip = pos.flipped();
      bside = false;
    }
    EncInfo* ei = first_ei(be, type);
    int p[TB_PIECES];
    size_t idx;
    int t = 0;
    uint8_t flags = 0;
    if (!be->hasPawns) {
      if (type == DTZ) {
        flags = PIECE(be)->dtzFlags;
        if ((flags & 1) != bside && !be->symmetric) {
          *success = -1;
          return 0;
        }
      }
      ei = type != DTZ ? &ei[bside] : ei;
      for (int i = 0; i < be->num;) {
        i = fill_squares(pos, ei->pieces, flip, 0, p, i);
      }
      idx = encode_piece(p, ei, be);
    } else {
      int i = fill_squares(pos, ei->pieces, flip, flip ? 0x38 : 0, p, 0);
      t = leading_pawn(p, be, type != DTM ? FILE_ENC : RANK_ENC);
      if (type == DTZ) {
        flags = PAWN(be)->dtzFlags[t];
        if ((flags & 1) != bside && !be->symmetric) {
          *success = -1;
          return 0;
        }
      }
      ei = type == WDL ? &ei[t + 4 * bside]
                       : type == DTM ? &ei[t + 6 * bside] : &ei[t];
      while (i < be->num) {
        i = fill_squares(pos, ei->pieces, flip, flip ? 0x38 : 0, p, i);
      }
      idx = type != DTM ? encode_pawn_f(p, ei, be) : encode_pawn_r(p, ei, be);
    }
    uint8_t* w = decompress_pairs(ei->precomp, idx);
    if (type == WDL) return static_cast<int>(w[0]) - 2;
    int v = w[0] + ((w[1] & 0x0f) << 8);
    if (type == DTM) {
      if (!be->dtmLossOnly) {
        v = from_le_u16(
            be->hasPawns
                ? PAWN(be)->dtmMap[PAWN(be)->dtmMapIdx[t][bside][s] + v]
                : PIECE(be)->dtmMap[PIECE(be)->dtmMapIdx[bside][s] + v]);
      }
    } else {
      if (flags & 2) {
        const int m = kWdlToMap[s + 2];
        if (!(flags & 16)) {
          v = be->hasPawns
                  ? static_cast<uint8_t*>(
                        PAWN(be)->dtzMap)[PAWN(be)->dtzMapIdx[t][m] + v]
                  : static_cast<uint8_t*>(
                        PIECE(be)->dtzMap)[PIECE(be)->dtzMapIdx[m] + v];
        } else {
          v = from_le_u16(
              be->hasPawns
                  ? static_cast<uint16_t*>(
                        PAWN(be)->dtzMap)[PAWN(be)->dtzMapIdx[t][m] + v]
                  : static_cast<uint16_t*>(
                        PIECE(be)->dtzMap)[PIECE(be)->dtzMapIdx[m] + v]);
        }
      }
      if (!(flags & kPAFlags[s + 2]) || (s & 1)) v *= 2;
    }
    return v;
  }
  int max_cardinality_ = 0;
  int max_cardinality_dtm_ = 0;
  Mutex ready_mutex_;
  std::string paths_;
  int num_piece_entries_ = 0;
  int num_pawn_entries_ = 0;
  int num_wdl_ = 0;
  int num_dtm_ = 0;
  int num_dtz_ = 0;
  std::vector<PieceEntry> piece_entries_;
  std::vector<PawnEntry> pawn_entries_;
  std::vector<TbHashEntry> tb_hash_;
};
SyzygyTablebase::SyzygyTablebase() : max_cardinality_(0) {}
SyzygyTablebase::~SyzygyTablebase() = default;
bool SyzygyTablebase::init(const std::string& paths) {
  paths_ = paths;
  impl_.reset(new SyzygyTablebaseImpl(paths_));
  max_cardinality_ = impl_->max_cardinality();
  if (max_cardinality_ <= 2) {
    impl_ = nullptr;
    return false;
  }
  return true;
}
// For a position where the side to move has a winning capture it is not
// necessary to store a winning value so the generator treats such positions as
// "don't cares" and tries to assign to it a value that improves the compression
// ratio. Similarly, if the side to move has a drawing capture, then the
// position is at least drawn. If the position is won, then the TB needs to
// store a win value. But if the position is drawn, the TB may store a loss
// value if that is better for compression. All of this means that during
// probing, the engine must look at captures and probe their results and must
// probe the position itself. The "best" result of these probes is the correct
// result for the position. DTZ table don't store values when a following move
// is a zeroing winning move (winning capture or winning pawn move). Also DTZ
// store wrong values for positions where the best move is an ep-move (even if
// losing). So in all these cases set the state to ZEROING_BEST_MOVE.
template <bool CheckZeroingMoves>
WDLScore SyzygyTablebase::search(const Position& pos, ProbeState* result) {
  WDLScore value;
  WDLScore best_value = WDL_LOSS;
  auto move_list = pos.GetBoard().GenerateLegalMoves();
  const size_t total_count = move_list.size();
  size_t move_count = 0;
  for (const Move& move : move_list) {
    if (!is_capture(pos.GetBoard(), move) &&
        (!CheckZeroingMoves || !pos.GetBoard().pawns().get(move.from()))) {
      continue;
    }
    move_count++;
    auto new_pos = Position(pos, move);
    value = static_cast<WDLScore>(-search(new_pos, result));
    if (*result == FAIL) return WDL_DRAW;
    if (value > best_value) {
      best_value = value;
      if (value >= WDL_WIN) {
        *result = ZEROING_BEST_MOVE;  // Winning DTZ-zeroing move
        return value;
      }
    }
  }
  // In case we have already searched all the legal moves we don't have to probe
  // the TB because the stored score could be wrong. For instance TB tables do
  // not contain information on position with ep rights, so in this case the
  // result of probe_wdl_table is wrong. Also in case of only capture moves, for
  // instance here 4K3/4q3/6p1/2k5/6p1/8/8/8 w - - 0 7, we have to return with
  // ZEROING_BEST_MOVE set.
  const bool no_more_moves = (move_count && move_count == total_count);
  if (no_more_moves) {
    value = best_value;
  } else {
    int raw_result = static_cast<int>(ProbeState::OK);
    value = static_cast<WDLScore>(
        impl_->probe_wdl_table(pos.GetBoard(), &raw_result));
    *result = static_cast<ProbeState>(raw_result);
    if (*result == FAIL) return WDL_DRAW;
  }
  // DTZ stores a "don't care" value if bestValue is a win
  if (best_value >= value) {
    *result = (best_value > WDL_DRAW || no_more_moves ? ZEROING_BEST_MOVE : OK);
    return best_value;
  }
  *result = OK;
  return value;
}
// Probe the WDL table for a particular position.
// If *result != FAIL, the probe was successful.
// The return value is from the point of view of the side to move:
// -2 : loss
// -1 : loss, but draw under 50-move rule
//  0 : draw
//  1 : win, but draw under 50-move rule
//  2 : win
WDLScore SyzygyTablebase::probe_wdl(const Position& pos, ProbeState* result) {
  *result = OK;
  return search(pos, result);
}
// Probe the DTZ table for a particular position.
// If *result != FAIL, the probe was successful.
// The return value is from the point of view of the side to move:
//         n < -100 : loss, but draw under 50-move rule
// -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
//        -1        : loss, the side to move is mated
//         0        : draw
//     1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
//   100 < n        : win, but draw under 50-move rule
//
// The return value n can be off by 1: a return value -n can mean a loss  in n+1
// ply and a return value +n can mean a win in n+1 ply. This cannot happen for
// tables with positions exactly on the "edge" of the 50-move rule.
//
// This implies that if dtz > 0 is returned, the position is certainly a win if
// dtz + 50-move-counter <= 99. Care must be taken that the engine picks moves
// that preserve dtz + 50-move-counter <= 99.
//
// If n = 100 immediately after a capture or pawn move, then the position is
// also certainly a win, and during the whole phase until the next capture or
// pawn move, the inequality to be preserved is dtz
// + 50-movecounter <= 100.
//
// In short, if a move is available resulting in dtz + 50-move-counter <= 99,
// then do not accept moves leading to dtz + 50-move-counter == 100.
int SyzygyTablebase::probe_dtz(const Position& pos, ProbeState* result) {
  *result = OK;
  const WDLScore wdl = search<true>(pos, result);
  if (*result == FAIL || wdl == WDL_DRAW) {  // DTZ tables don't store draws
    return 0;
  }
  // DTZ stores a 'don't care' value in this case, or even a plain wrong one as
  // in case the best move is a losing ep, so it cannot be probed.
  if (*result == ZEROING_BEST_MOVE) return dtz_before_zeroing(wdl);
  int raw_result = 1;
  int dtz = impl_->probe_dtz_table(pos.GetBoard(), wdl, &raw_result);
  *result = static_cast<ProbeState>(raw_result);
  if (*result == FAIL) return 0;
  if (*result != CHANGE_STM) {
    return (dtz + 1 +
            100 * (wdl == WDL_BLESSED_LOSS || wdl == WDL_CURSED_WIN)) *
           sign_of(wdl);
  }
  // DTZ stores results for the other side, so we need to do a 1-ply search and
  // find the winning move that minimizes DTZ.
  int min_DTZ = 0xFFFF;
  for (const Move& move : pos.GetBoard().GenerateLegalMoves()) {
    Position next_pos = Position(pos, move);
    const bool zeroing = next_pos.GetRule50Ply() == 0;
    // For zeroing moves we want the dtz of the move _before_ doing it,
    // otherwise we will get the dtz of the next move sequence. Search the
    // position after the move to get the score sign (because even in a winning
    // position we could make a losing capture or going for a draw).
    dtz = zeroing ? -dtz_before_zeroing(search(next_pos, result))
                  : -probe_dtz(next_pos, result);
    // If the move mates, force minDTZ to 1
    if (dtz == 1 && next_pos.GetBoard().IsUnderCheck() &&
        next_pos.GetBoard().GenerateLegalMoves().empty()) {
      min_DTZ = 1;
    }
    // Convert result from 1-ply search. Zeroing moves are already accounted by
    // dtz_before_zeroing() that returns the DTZ of the previous move.
    if (!zeroing) dtz += sign_of(dtz);
    // Skip the draws and if we are winning only pick positive dtz
    if (dtz < min_DTZ && sign_of(dtz) == sign_of(wdl)) min_DTZ = dtz;
    if (*result == FAIL) return 0;
  }
  // When there are no legal moves, the position is mate: we return -1
  return min_DTZ == 0xFFFF ? -1 : min_DTZ;
}
// Use the DTZ tables to rank root moves.
//
// A return value false indicates that not all probes were successful.
bool SyzygyTablebase::root_probe(const Position& pos, bool has_repeated,
                                 std::vector<Move>* safe_moves) {
  ProbeState result;
  auto root_moves = pos.GetBoard().GenerateLegalMoves();
  // Obtain 50-move counter for the root position
  const int cnt50 = pos.GetRule50Ply();
  // Check whether a position was repeated since the last zeroing move.
  const bool rep = has_repeated;
  int dtz;
  std::vector<int> ranks;
  ranks.reserve(root_moves.size());
  int best_rank = -1000;
  // Probe and rank each move
  for (auto& m : root_moves) {
    Position next_pos = Position(pos, m);
    // Calculate dtz for the current move counting from the root position
    if (next_pos.GetRule50Ply() == 0) {
      // In case of a zeroing move, dtz is one of -101/-1/0/1/101
      const WDLScore wdl = static_cast<WDLScore>(-probe_wdl(next_pos, &result));
      dtz = dtz_before_zeroing(wdl);
    } else {
      // Otherwise, take dtz for the new position and correct by 1 ply
      dtz = -probe_dtz(next_pos, &result);
      dtz = dtz > 0 ? dtz + 1 : dtz < 0 ? dtz - 1 : dtz;
    }
    // Make sure that a mating move is assigned a dtz value of 1
    if (next_pos.GetBoard().IsUnderCheck() && dtz == 2 &&
        next_pos.GetBoard().GenerateLegalMoves().size() == 0) {
      dtz = 1;
    }
    if (result == FAIL) return false;
    // Better moves are ranked higher. Certain wins are ranked equally.
    // Losing moves are ranked equally unless a 50-move draw is in sight.
    int r = dtz > 0
                ? (dtz + cnt50 <= 99 && !rep ? 1000 : 1000 - (dtz + cnt50))
                : dtz < 0 ? (-dtz * 2 + cnt50 < 100 ? -1000
                                                    : -1000 + (-dtz + cnt50))
                          : 0;
    if (r > best_rank) best_rank = r;
    ranks.push_back(r);
  }
  // Disable all but the equal best moves.
  int counter = 0;
  for (auto& m : root_moves) {
    if (ranks[counter] == best_rank) {
      safe_moves->push_back(m);
    }
    counter++;
  }
  return true;
}
// Use the WDL tables to rank root moves.
// This is a fallback for the case that some or all DTZ tables are missing.
//
// A return value false indicates that not all probes were successful.
bool SyzygyTablebase::root_probe_wdl(const Position& pos,
                                     std::vector<Move>* safe_moves) {
  static const int WDL_to_rank[] = {-1000, -899, 0, 899, 1000};
  auto root_moves = pos.GetBoard().GenerateLegalMoves();
  ProbeState result;
  std::vector<int> ranks;
  ranks.reserve(root_moves.size());
  int best_rank = -1000;
  // Probe and rank each move
  for (auto& m : root_moves) {
    Position nextPos = Position(pos, m);
    const WDLScore wdl = static_cast<WDLScore>(-probe_wdl(nextPos, &result));
    if (result == FAIL) return false;
    ranks.push_back(WDL_to_rank[wdl + 2]);
    if (ranks.back() > best_rank) best_rank = ranks.back();
  }
  // Disable all but the equal best moves.
  int counter = 0;
  for (auto& m : root_moves) {
    if (ranks[counter] == best_rank) {
      safe_moves->push_back(m);
    }
    counter++;
  }
  return true;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/syzygy/syzygy.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/numa.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#ifdef _WIN32
#endif
namespace lczero {
int Numa::threads_per_core_ = 1;
void Numa::Init() {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* buffer;
  DWORD len;
  GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
  buffer = static_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(malloc(len));
  GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &len);
  if (buffer->Processor.Flags & LTP_PC_SMT) {
    threads_per_core_ = BitBoard(buffer->Processor.GroupMask[0].Mask).count();
  }
  free(buffer);
  int group_count = GetActiveProcessorGroupCount();
  int thread_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  int core_count = thread_count / threads_per_core_;
  CERR << "Detected " << core_count << " core(s) and " << thread_count
       << " thread(s) in " << group_count << " group(s).";
  for (int group_id = 0; group_id < group_count; group_id++) {
    int group_threads = GetActiveProcessorCount(group_id);
    int group_cores = group_threads / threads_per_core_;
    CERR << "Group " << group_id << " has " << group_cores
         << " core(s) and " << group_threads << " thread(s).";
  }
#endif
}
void Numa::BindThread(int id) {
#if defined(_WIN64) && _WIN32_WINNT >= 0x0601
  int group_count = GetActiveProcessorGroupCount();
  int thread_count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  int core_count = thread_count / threads_per_core_;
  int core_id = id;
  GROUP_AFFINITY affinity = {};
  for (int group_id = 0; group_id < group_count; group_id++) {
    int group_threads = GetActiveProcessorCount(group_id);
    int group_cores = group_threads / threads_per_core_;
    // Allocate cores of each group in order, and distribute remaining threads
    // to all groups.
    if ((id < core_count && core_id < group_cores) ||
        (id >= core_count && (id - core_count) % group_count == group_id)) {
      affinity.Group = group_id;
      affinity.Mask = ~0ULL >> (64 - group_threads);
      SetThreadGroupAffinity(GetCurrentThread(), &affinity, NULL);
      break;
    }
    core_id -= group_cores;
  }
#else
  // Silence warning.
  (void)id;
#endif
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/numa.cc

// begin of /Users/syys/CLionProjects/lc0/src/mcts/search.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}
class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}
  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{true},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}
  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }
  float GetM(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child.GetM(parent_m_);
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }
  float GetM(Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }
  // The M utility to use for unvisited nodes.
  float GetDefaultM() const { return 0.0f; }
 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }
  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};
}  // namespace
Search::Search(const NodeTree& tree, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
}
namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;
  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }
  if (total < std::numeric_limits<float>::min()) return;
  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace
void Search::SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false);
  std::vector<ThinkingInfo> uci_infos;
  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (display_cache_usage) {
    common_info.hashfull =
        cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_playouts_ * 1000 / time_since_first_batch_ms;
    }
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);
  int multipv = 0;
  const auto default_q = -root_node_->GetQ(-draw_score);
  const auto default_wl = -root_node_->GetWL();
  const auto default_d = root_node_->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    const auto wl = edge.GetWL(default_wl);
    const auto floatD = edge.GetD(default_d);
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f)) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    }
    auto w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - floatD))));
    auto l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - floatD))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto d = 1000 - w - l;
    if (d < 0) {
      w = std::min(1000, std::max(0, w + d / 2));
      l = 1000 - w;
      d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{w, d, l};
    if (network_->GetCapabilities().has_mlh()) {
      uci_info.moves_left = static_cast<int>(
          (1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
      depth += 1;
    }
  }
  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    last_outputted_info_edge_ = current_best_edge_.edge();
  }
  uci_responder_->OutputThinkingInfo(&uci_infos);
}
// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
}
int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}
int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}
// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth ? params_.GetOpponentDrawScore()
                       : params_.GetSidetomoveDrawScore()) +
         (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetWhiteDrawDelta()
              : params_.GetBlackDrawDelta());
}
namespace {
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) -
                   value * std::sqrt(node->GetVisitedPolicy());
}
// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) - value * std::sqrt(visited_pol);
}
inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace
std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) edges.push_back(edge);
  std::sort(edges.begin(), edges.end(),
            [&fpu, &U_coeff, &draw_score](EdgeAndNode a, EdgeAndNode b) {
              return std::forward_as_tuple(
                         a.GetN(), a.GetQ(fpu, draw_score) + a.GetU(U_coeff)) <
                     std::forward_as_tuple(
                         b.GetN(), b.GetQ(fpu, draw_score) + b.GetU(U_coeff));
            });
  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      print(oss, "(WL: ", sign * n->GetWL(), ") ", 8, 5);
      print(oss, "(D: ", n->GetD(), ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
    }
    print(oss, "(Q: ", n ? sign * n->GetQ(sign * draw_score) : fpu, ") ", 8, 5);
  };
  auto print_tail = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    std::optional<float> v;
    if (n && n->IsTerminal()) {
      v = n->GetQ(sign * draw_score);
    } else {
      NNCacheLock nneval = GetCachedNNEval(n);
      if (nneval) v = -nneval->q;
    }
    if (v) {
      print(oss, "(V: ", sign * *v, ") ", 7, 4);
    } else {
      *oss << "(V:  -.----) ";
    }
    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up                                                ? "(T) "
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) "
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) "
                                                                       : "");
    }
  };
  std::vector<std::string> infos;
  const auto m_evaluator = network_->GetCapabilities().has_mlh()
                               ? MEvaluator(params_, node)
                               : MEvaluator();
  for (const auto& edge : edges) {
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetM(edge, Q);
    std::ostringstream oss;
    oss << std::left;
    // TODO: should this be displaying transformed index?
    print_head(&oss, edge.GetMove(is_black_to_move).as_string(),
               edge.GetMove().as_nn_index(0), edge.GetN(), edge.GetNInFlight(),
               edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", edge.GetU(U_coeff), ") ", 6, 5);
    print(&oss, "(S: ", Q + edge.GetU(U_coeff) + M, ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }
  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);
  infos.emplace_back(oss.str());
  return infos;
}
void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);
  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  for (auto& edge : root_node_->Edges()) {
    if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: " << final_bestmove_.as_string();
      for (const auto& line : GetVerboseStats(edge.node())) {
        LOGFILE << line;
      }
    }
  }
}
NNCacheLock Search::GetCachedNNEval(const Node* node) const {
  if (!node) return {};
  std::vector<Move> moves;
  for (; node != root_node_; node = node->GetParent()) {
    moves.push_back(node->GetOwnEdge()->GetMove());
  }
  PositionHistory history(played_history_);
  for (auto iter = moves.rbegin(), end = moves.rend(); iter != end; ++iter) {
    history.Append(*iter);
  }
  const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}
void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ + initial_visits_ == 0) return;
  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }
  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  }
  // Use a 0 visit cancel score update to clear out any cached best edge, as
  // at the next iteration remaining playouts may be different.
  // TODO(crem) Is it really needed?
  root_node_->CancelScoreUpdate(0);
}
// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);
  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.IsTerminal();
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}
std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}
std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}
void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}
// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (root_node_->GetN() == 0) return;
  if (!root_node_->HasChildren()) return;
  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;
  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && decay_moves) {
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }
  auto bestmove_edge = temperature
                           ? GetBestRootChildWithTemperature(temperature)
                           : GetBestChildNoTemperature(root_node_, 0);
  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());
  if (bestmove_edge.GetN() > 0 && bestmove_edge.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1)
                            .GetMove(!played_history_.IsBlackToMove());
  }
}
// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  // Even if Edges is populated at this point, its a race condition to access
  // the node, so exit quickly.
  if (parent->GetN() == 0) return {};
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    if (parent == root_node_ && !root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    edges.push_back(edge);
  }
  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();
  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [draw_score](const auto& a, const auto& b) {
        // The function returns "true" when a is preferred to b.
        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };
        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // This default isn't used as wl only checked for case edge is
          // terminal.
          const auto wl = edge.GetWL(0.0f);
          // Not safe to access IsTerminal if GetN is 0.
          if (edge.GetN() == 0 || !edge.IsTerminal() || !wl) {
            return kNonTerminal;
          }
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };
        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;
        // If both are terminal draws, try to make it shorter.
        // Not safe to access IsTerminal if GetN is 0.
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }
        // Neither is terminal, use standard rule.
        if (a_rank == kNonTerminal) {
          // Prefer largest playouts then eval then prior.
          if (a.GetN() != b.GetN()) return a.GetN() > b.GetN();
          // Default doesn't matter here so long as they are the same as either
          // both are N==0 (thus we're comparing equal defaults) or N!=0 and
          // default isn't used.
          if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) {
            return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
          }
          return a.GetP() > b.GetP();
        }
        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }
        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
      });
  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}
// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}
// Returns a child of a root chosen according to weighted-by-temperature visit
// count.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);
  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) {
      max_n = edge.GetN() + offset;
      max_eval = edge.GetQ(fpu, draw_score);
    }
  }
  // No move had enough visits for temperature, so use default child criteria
  if (max_n <= 0.0f) return GetBestChildNoTemperature(root_node_, 0);
  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    sum += std::pow(
        std::max(0.0f, (static_cast<float>(edge.GetN()) + offset) / max_n),
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  assert(sum);
  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    if (idx-- == 0) return edge;
  }
  assert(false);
  return {};
}
void Search::StartThreads(size_t how_many) {
  thread_count_.store(how_many, std::memory_order_release);
  Mutex::Lock lock(threads_mutex_);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this, i]() {
      SearchWorker worker(this, params_, i);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}
void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}
bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}
void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();
  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  stats->total_nodes = total_playouts_ + initial_visits_;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->win_found = false;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true);
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -1000;
    uint64_t max_n = 0;
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = network_->GetCapabilities().has_mlh()
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge : root_node_->Edges()) {
      const auto n = edge.GetN();
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetM(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) {
        stats->win_found = true;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
        stats->num_losing_edges += 1;
      }
      if (max_n < n) {
        max_n = n;
        max_n_has_max_q_plus_m = false;
      }
      if (max_q_plus_m <= q_plus_m) {
        max_n_has_max_q_plus_m = (max_n == n);
        max_q_plus_m = q_plus_m;
      }
    }
    if (!max_n_has_max_q_plus_m) {
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}
void Search::WatchdogThread() {
  Numa::BindThread(0);
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    hints.Reset();
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo();
    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;
    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;
    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}
void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}
void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}
void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}
void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}
void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    Node* node = entry.first;
    for (node = node->GetParent(); node != root_node_->GetParent();
         node = node->GetParent()) {
      node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}
Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();
  }
  LOGFILE << "Search destroyed.";
}
//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////
void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    {
      int spins = 0;
      while (true) {
        int nta = tasks_taken_.load(std::memory_order_acquire);
        int tc = task_count_.load(std::memory_order_acquire);
        if (nta < tc) {
          int val = 0;
          if (task_taking_started_.compare_exchange_weak(
                  val, 1, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            nta = tasks_taken_.load(std::memory_order_acquire);
            tc = task_count_.load(std::memory_order_acquire);
            // We got the spin lock, double check we're still in the clear.
            if (nta < tc) {
              id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
              task = &picking_tasks_[id];
              task_taking_started_.store(0, std::memory_order_release);
              break;
            }
            task_taking_started_.store(0, std::memory_order_release);
          }
          SpinloopPause();
          spins = 0;
          continue;
        } else if (tc != -1) {
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        spins = 0;
        // Looks like sleep time.
        Mutex::Lock lock(picking_tasks_mutex_);
        // Refresh them now we have the lock.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (tc != -1) continue;
        if (nta >= tc && exiting_) return;
        task_added_.wait(lock.get_raw());
        // And refresh again now we're awake.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (nta >= tc && exiting_) return;
      }
    }
    if (task != nullptr) {
      switch (task->task_type) {
        case PickTask::kGathering: {
          PickNodesToExtendTask(task->start, task->base_depth,
                                task->collision_limit, task->moves_to_base,
                                &(task->results), &(task_workspaces_[tid]));
          break;
        }
        case PickTask::kProcessing: {
          ProcessPickedTask(task->start_idx, task->end_idx,
                            &(task_workspaces_[tid]));
          break;
        }
      }
      picking_tasks_[id].complete = true;
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}
void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->NewComputation());
  if (params_.GetMaxConcurrentSearchers() != 0) {
    while (true) {
      // If search is stop, we've not gathered or done anything and we don't
      // want to, so we can safely skip all below. But make sure we have done
      // at least one iteration.
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }
      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available > 0 &&
          search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      }
      // This is a hard spin lock to reduce latency but at the expense of busy
      // wait cpu usage. If search worker count is large, this is probably a bad
      // idea.
    }
  }
  // 2. Gather minibatch.
  GatherMinibatch2();
  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);
  // 2b. Collect collisions.
  CollectCollisions();
  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();
  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }
  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);
  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();
  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();
  // 7. Update the Search's status and progress information.
  UpdateCounters();
  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      {
        Mutex::Lock lock(search_->counters_mutex_);
        time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      }
      if (time_since_first_batch_ms <= 0) {
        time_since_first_batch_ms = search_->GetTimeSinceStart();
      }
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}
// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::make_unique<CachingComputation>(std::move(computation),
                                                      search_->cache_);
  computation_->Reserve(params_.GetMiniBatchSize());
  minibatch_.clear();
  minibatch_.reserve(2 * params_.GetMiniBatchSize());
}
// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
namespace {
int Mix(int high, int low, float ratio) {
  return static_cast<int>(std::round(static_cast<float>(low) +
                                     static_cast<float>(high - low) * ratio));
}
int CalculateCollisionsLeft(int64_t nodes, const SearchParams& params) {
  // End checked first
  if (nodes >= params.GetMaxCollisionVisitsScalingEnd()) {
    return params.GetMaxCollisionVisits();
  }
  if (nodes <= params.GetMaxCollisionVisitsScalingStart()) {
    return 1;
  }
  return Mix(params.GetMaxCollisionVisits(), 1,
             std::pow((static_cast<float>(nodes) -
                       params.GetMaxCollisionVisitsScalingStart()) /
                          (params.GetMaxCollisionVisitsScalingEnd() -
                           params.GetMaxCollisionVisitsScalingStart()),
                      params.GetMaxCollisionVisitsScalingPower()));
}
}  // namespace
void SearchWorker::GatherMinibatch2() {
  // Total number of nodes to process.
  int minibatch_size = 0;
  int cur_n = 0;
  {
    SharedMutex::Lock lock(search_->nodes_mutex_);
    cur_n = search_->root_node_->GetN();
  }
  // TODO: GetEstimatedRemainingPlayouts has already had smart pruning factor
  // applied, which doesn't clearly make sense to include here...
  int64_t remaining_n =
      latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
  int collisions_left = CalculateCollisionsLeft(
      std::min(static_cast<int64_t>(cur_n), remaining_n), params_);
  // Number of nodes processed out of order.
  number_out_of_order_ = 0;
  int thread_count = search_->thread_count_.load(std::memory_order_acquire);
  // Gather nodes to process in the current batch.
  // If we had too many nodes out of order, also interrupt the iteration so
  // that search can exit.
  while (minibatch_size < params_.GetMiniBatchSize() &&
         number_out_of_order_ < params_.GetMaxOutOfOrderEvals()) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;
    // If there is backend work to be done, and the backend is idle - exit
    // immediately.
    // Only do this fancy work if there are multiple threads as otherwise we
    // early exit from every batch since there is never another search thread to
    // be keeping the backend busy. Which would mean that threads=1 has a
    // massive nps drop.
    if (thread_count > 1 && minibatch_size > 0 &&
        computation_->GetCacheMisses() > params_.GetIdlingMinimumWork() &&
        thread_count - search_->backend_waiting_counter_.load(
                           std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }
    int new_start = static_cast<int>(minibatch_.size());
    PickNodesToExtend(
        std::min({collisions_left, params_.GetMiniBatchSize() - minibatch_size,
                  params_.GetMaxOutOfOrderEvals() - number_out_of_order_}));
    // Count the non-collisions.
    int non_collisions = 0;
    for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        continue;
      }
      ++non_collisions;
      ++minibatch_size;
    }
    bool needs_wait = false;
    int ppt_start = new_start;
    if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
        non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
      const int num_tasks = std::clamp(
          non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2,
          params_.GetTaskWorkersPerSearchWorker() + 1);
      // Round down, left overs can go to main thread so it waits less.
      int per_worker = non_collisions / num_tasks;
      needs_wait = true;
      ResetTasks();
      int found = 0;
      for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
        auto& picked_node = minibatch_[i];
        if (picked_node.IsCollision()) {
          continue;
        }
        ++found;
        if (found == per_worker) {
          picking_tasks_.emplace_back(ppt_start, i + 1);
          task_count_.fetch_add(1, std::memory_order_acq_rel);
          ppt_start = i + 1;
          found = 0;
          if (picking_tasks_.size() == static_cast<size_t>(num_tasks - 1)) {
            break;
          }
        }
      }
    }
    ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()),
                      &main_workspace_);
    if (needs_wait) {
      WaitForTasks();
    }
    bool some_ooo = false;
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
      if (minibatch_[i].ooo_completed) {
        some_ooo = true;
        break;
      }
    }
    if (some_ooo) {
      SharedMutex::Lock lock(search_->nodes_mutex_);
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start;
           i--) {
        // If there was any OOO, revert 'all' new collisions - it isn't possible
        // to identify exactly which ones are afterwards and only prune those.
        // This may remove too many items, but hopefully most of the time they
        // will just be added back in the same in the next gather.
        if (minibatch_[i].IsCollision()) {
          Node* node = minibatch_[i].node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
            node->CancelScoreUpdate(minibatch_[i].multivisit);
          }
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --minibatch_size;
          ++number_out_of_order_;
        }
      }
    }
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      // If there was no OOO, there can stil be collisions.
      // There are no OOO though.
      // Also terminals when OOO is disabled.
      if (!minibatch_[i].nn_queried) continue;
      if (minibatch_[i].is_cache_hit) {
        // Since minibatch_[i] holds cache lock, this is guaranteed to succeed.
        computation_->AddInputByHash(minibatch_[i].hash,
                                     std::move(minibatch_[i].lock));
      } else {
        computation_->AddInput(minibatch_[i].hash,
                               std::move(minibatch_[i].input_planes),
                               std::move(minibatch_[i].probabilities_to_cache));
      }
    }
    // Check for stop at the end so we have at least one node.
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        // Check to see if we can upsize the collision to exit sooner.
        if (picked_node.maxvisit > 0 &&
            collisions_left > picked_node.multivisit) {
          SharedMutex::Lock lock(search_->nodes_mutex_);
          int extra = std::min(picked_node.maxvisit, collisions_left) -
                      picked_node.multivisit;
          picked_node.multivisit += extra;
          Node* node = picked_node.node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
            node->IncrementNInFlight(extra);
          }
        }
        if ((collisions_left -= picked_node.multivisit) <= 0) return;
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
  }
}
void SearchWorker::ProcessPickedTask(int start_idx, int end_idx,
                                     TaskWorkspace* workspace) {
  auto& history = workspace->history;
  history = search_->played_history_;
  for (int i = start_idx; i < end_idx; i++) {
    auto& picked_node = minibatch_[i];
    if (picked_node.IsCollision()) continue;
    auto* node = picked_node.node;
    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), it means that we already visited this node before.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(node, picked_node.depth, picked_node.moves_to_visit, &history);
      if (!node->IsTerminal()) {
        picked_node.nn_queried = true;
        const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
        picked_node.hash = hash;
        picked_node.lock = NNCacheLock(search_->cache_, hash);
        picked_node.is_cache_hit = picked_node.lock;
        if (!picked_node.is_cache_hit) {
          int transform;
          picked_node.input_planes = EncodePositionForNN(
              search_->network_->GetCapabilities().input_format, history, 8,
              params_.GetHistoryFill(), &transform);
          picked_node.probability_transform = transform;
          std::vector<uint16_t>& moves = picked_node.probabilities_to_cache;
          // Legal moves are known, use them.
          moves.reserve(node->GetNumEdges());
          for (const auto& edge : node->Edges()) {
            moves.emplace_back(edge.GetMove().as_nn_index(transform));
          }
        } else {
          picked_node.probability_transform = TransformForPosition(
              search_->network_->GetCapabilities().input_format, history);
        }
      }
    }
    if (params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder()) {
      // Perform out of order eval for the last entry in minibatch_.
      FetchSingleNodeResult(&picked_node, picked_node, 0);
      picked_node.ooo_completed = true;
    }
  }
}
#define MAX_TASKS 100
void SearchWorker::ResetTasks() {
  task_count_.store(0, std::memory_order_release);
  tasks_taken_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  picking_tasks_.clear();
  // Reserve because resizing breaks pointers held by the task threads.
  picking_tasks_.reserve(MAX_TASKS);
}
int SearchWorker::WaitForTasks() {
  // Spin lock, other tasks should be done soon.
  while (true) {
    int completed = completed_tasks_.load(std::memory_order_acquire);
    int todo = task_count_.load(std::memory_order_acquire);
    if (todo == completed) return completed;
    SpinloopPause();
  }
}
void SearchWorker::PickNodesToExtend(int collision_limit) {
  ResetTasks();
  {
    // While nothing is ready yet - wake the task runners so they are ready to
    // receive quickly.
    Mutex::Lock lock(picking_tasks_mutex_);
    task_added_.notify_all();
  }
  std::vector<Move> empty_movelist;
  // This lock must be held until after the task_completed_ wait succeeds below.
  // Since the tasks perform work which assumes they have the lock, even though
  // actually this thread does.
  SharedMutex::Lock lock(search_->nodes_mutex_);
  PickNodesToExtendTask(search_->root_node_, 0, collision_limit, empty_movelist,
                        &minibatch_, &main_workspace_);
  WaitForTasks();
  for (int i = 0; i < static_cast<int>(picking_tasks_.size()); i++) {
    for (int j = 0; j < static_cast<int>(picking_tasks_[i].results.size());
         j++) {
      minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
    }
  }
}
void SearchWorker::EnsureNodeTwoFoldCorrectForDepth(Node* child_node,
                                                    int depth) {
  // Check whether first repetition was before root. If yes, remove
  // terminal status of node and revert all visits in the tree.
  // Length of repetition was stored in m_. This code will only do
  // something when tree is reused and twofold visits need to be
  // reverted.
  if (child_node->IsTwoFoldTerminal() && depth < child_node->GetM()) {
    // Take a mutex - any SearchWorker specific mutex... since this is
    // not safe to do concurrently between multiple tasks.
    Mutex::Lock lock(picking_tasks_mutex_);
    int depth_counter = 0;
    // Cache node's values as we reset them in the process. We could
    // manually set wl and d, but if we want to reuse this for reverting
    // other terminal nodes this is the way to go.
    const auto wl = child_node->GetWL();
    const auto d = child_node->GetD();
    const auto m = child_node->GetM();
    const auto terminal_visits = child_node->GetN();
    for (Node* node_to_revert = child_node; node_to_revert != nullptr;
         node_to_revert = node_to_revert->GetParent()) {
      // Revert all visits on twofold draw when making it non terminal.
      node_to_revert->RevertTerminalVisits(wl, d, m + (float)depth_counter,
                                           terminal_visits);
      depth_counter++;
      // Even if original tree still exists, we don't want to revert
      // more than until new root.
      if (depth_counter > depth) break;
      // If wl != 0, we would have to switch signs at each depth.
    }
    // Mark the prior twofold draw as non terminal to extend it again.
    child_node->MakeNotTerminal();
    // When reverting the visits, we also need to revert the initial
    // visits, as we reused fewer nodes than anticipated.
    search_->initial_visits_ -= terminal_visits;
    // Max depth doesn't change when reverting the visits, and
    // cum_depth_ only counts the average depth of new nodes, not reused
    // ones.
  }
}
void SearchWorker::PickNodesToExtendTask(Node* node, int base_depth,
                                         int collision_limit,
                                         const std::vector<Move>& moves_to_base,
                                         std::vector<NodeToProcess>* receiver,
                                         TaskWorkspace* workspace) {
  // TODO: Bring back pre-cached nodes created outside locks in a way that works
  // with tasks.
  // TODO: pre-reserve visits_to_perform for expected depth and likely maximum
  // width. Maybe even do so outside of lock scope.
  auto& vtp_buffer = workspace->vtp_buffer;
  auto& visits_to_perform = workspace->visits_to_perform;
  visits_to_perform.clear();
  auto& vtp_last_filled = workspace->vtp_last_filled;
  vtp_last_filled.clear();
  auto& current_path = workspace->current_path;
  current_path.clear();
  auto& moves_to_path = workspace->moves_to_path;
  moves_to_path = moves_to_base;
  // Sometimes receiver is reused, othertimes not, so only jump start if small.
  if (receiver->capacity() < 30) {
    receiver->reserve(receiver->size() + 30);
  }
  // These 2 are 'filled pre-emptively'.
  std::array<float, 256> current_pol;
  std::array<float, 256> current_util;
  // These 3 are 'filled on demand'.
  std::array<float, 256> current_score;
  std::array<int, 256> current_nstarted;
  auto& cur_iters = workspace->cur_iters;
  Node::Iterator best_edge;
  Node::Iterator second_best_edge;
  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();
  int passed_off = 0;
  int completed_visits = 0;
  bool is_root_node = node == search_->root_node_;
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);
  const auto& root_move_filter = search_->root_move_filter_;
  auto m_evaluator = moves_left_support_ ? MEvaluator(params_) : MEvaluator();
  int max_limit = std::numeric_limits<int>::max();
  current_path.push_back(-1);
  while (current_path.size() > 0) {
    // First prepare visits_to_perform.
    if (current_path.back() == -1) {
      // Need to do n visits, where n is either collision_limit, or comes from
      // visits_to_perform for the current path.
      int cur_limit = collision_limit;
      if (current_path.size() > 1) {
        cur_limit =
            (*visits_to_perform.back())[current_path[current_path.size() - 2]];
      }
      // First check if node is terminal or not-expanded.  If either than create
      // a collision of appropriate size and pop current_path.
      if (node->GetN() == 0 || node->IsTerminal()) {
        if (is_root_node) {
          // Root node is special - since its not reached from anywhere else, so
          // it needs its own logic. Still need to create the collision to
          // ensure the outer gather loop gives up.
          if (node->TryStartScoreUpdate()) {
            cur_limit -= 1;
            minibatch_.push_back(NodeToProcess::Visit(
                node, static_cast<uint16_t>(current_path.size() + base_depth)));
            completed_visits++;
          }
        }
        // Visits are created elsewhere, just need the collisions here.
        if (cur_limit > 0) {
          int max_count = 0;
          if (cur_limit == collision_limit && base_depth == 0 &&
              max_limit > cur_limit) {
            max_count = max_limit;
          }
          receiver->push_back(NodeToProcess::Collision(
              node, static_cast<uint16_t>(current_path.size() + base_depth),
              cur_limit, max_count));
          completed_visits += cur_limit;
        }
        node = node->GetParent();
        current_path.pop_back();
        continue;
      }
      if (is_root_node) {
        // Root node is again special - needs its n in flight updated separately
        // as its not handled on the path to it, since there isn't one.
        node->IncrementNInFlight(cur_limit);
      }
      // Create visits_to_perform new back entry for this level.
      if (vtp_buffer.size() > 0) {
        visits_to_perform.push_back(std::move(vtp_buffer.back()));
        vtp_buffer.pop_back();
      } else {
        visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled.push_back(-1);
      // Cache all constant UCT parameters.
      // When we're near the leaves we can copy less of the policy, since there
      // is no way iteration will ever reach it.
      // TODO: This is a very conservative formula. It assumes every visit we're
      // aiming to add is going to trigger a new child, and that any visits
      // we've already had have also done so and then a couple extra since we go
      // to 2 unvisited to get second best in worst case.
      // Unclear we can do better without having already walked the children.
      // Which we are putting off until after policy is copied so we can create
      // visited policy without having to cache it in the node (allowing the
      // node to stay at 64 bytes).
      int max_needed = node->GetNumEdges();
      if (!is_root_node || root_move_filter.empty()) {
        max_needed = std::min(max_needed, node->GetNStarted() + cur_limit + 2);
      }
      node->CopyPolicy(max_needed, current_pol.data());
      for (int i = 0; i < max_needed; i++) {
        current_util[i] = std::numeric_limits<float>::lowest();
      }
      // Root depth is 1 here, while for GetDrawScore() it's 0-based, that's why
      // the weirdness.
      const float draw_score = ((current_path.size() + base_depth) % 2 == 0)
                                   ? odd_draw_score
                                   : even_draw_score;
      m_evaluator.SetParent(node);
      float visited_pol = 0.0f;
      for (Node* child : node->VisitedNodes()) {
        int index = child->Index();
        visited_pol += current_pol[index];
        float q = child->GetQ(draw_score);
        current_util[index] = q + m_evaluator.GetM(child, q);
      }
      const float fpu =
          GetFpu(params_, node, is_root_node, draw_score, visited_pol);
      for (int i = 0; i < max_needed; i++) {
        if (current_util[i] == std::numeric_limits<float>::lowest()) {
          current_util[i] = fpu + m_evaluator.GetDefaultM();
        }
      }
      const float cpuct = ComputeCpuct(params_, node->GetN(), is_root_node);
      const float puct_mult =
          cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
      int cache_filled_idx = -1;
      while (cur_limit > 0) {
        // Perform UCT for current node.
        float best = std::numeric_limits<float>::lowest();
        int best_idx = -1;
        float best_without_u = std::numeric_limits<float>::lowest();
        float second_best = std::numeric_limits<float>::lowest();
        bool can_exit = false;
        best_edge.Reset();
        for (int idx = 0; idx < max_needed; ++idx) {
          if (idx > cache_filled_idx) {
            if (idx == 0) {
              cur_iters[idx] = node->Edges();
            } else {
              cur_iters[idx] = cur_iters[idx - 1];
              ++cur_iters[idx];
            }
            current_nstarted[idx] = cur_iters[idx].GetNStarted();
          }
          int nstarted = current_nstarted[idx];
          const float util = current_util[idx];
          if (idx > cache_filled_idx) {
            current_score[idx] =
                current_pol[idx] * puct_mult / (1 + nstarted) + util;
            cache_filled_idx++;
          }
          if (is_root_node) {
            // If there's no chance to catch up to the current best node with
            // remaining playouts, don't consider it.
            // best_move_node_ could have changed since best_node_n was
            // retrieved. To ensure we have at least one node to expand, always
            // include current best node.
            if (cur_iters[idx] != search_->current_best_edge_ &&
                latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                    best_node_n - cur_iters[idx].GetN()) {
              continue;
            }
            // If root move filter exists, make sure move is in the list.
            if (!root_move_filter.empty() &&
                std::find(root_move_filter.begin(), root_move_filter.end(),
                          cur_iters[idx].GetMove()) == root_move_filter.end()) {
              continue;
            }
          }
          float score = current_score[idx];
          if (score > best) {
            second_best = best;
            second_best_edge = best_edge;
            best = score;
            best_idx = idx;
            best_without_u = util;
            best_edge = cur_iters[idx];
          } else if (score > second_best) {
            second_best = score;
            second_best_edge = cur_iters[idx];
          }
          if (can_exit) break;
          if (nstarted == 0) {
            // One more loop will get 2 unvisited nodes, which is sufficient to
            // ensure second best is correct. This relies upon the fact that
            // edges are sorted in policy decreasing order.
            can_exit = true;
          }
        }
        int new_visits = 0;
        if (second_best_edge) {
          int estimated_visits_to_change_best = std::numeric_limits<int>::max();
          if (best_without_u < second_best) {
            const auto n1 = current_nstarted[best_idx] + 1;
            estimated_visits_to_change_best = static_cast<int>(
                std::max(1.0f, std::min(current_pol[best_idx] * puct_mult /
                                                (second_best - best_without_u) -
                                            n1 + 1,
                                        1e9f)));
          }
          second_best_edge.Reset();
          max_limit = std::min(max_limit, estimated_visits_to_change_best);
          new_visits = std::min(cur_limit, estimated_visits_to_change_best);
        } else {
          // No second best - only one edge, so everything goes in here.
          new_visits = cur_limit;
        }
        if (best_idx >= vtp_last_filled.back()) {
          auto* vtp_array = visits_to_perform.back().get()->data();
          std::fill(vtp_array + (vtp_last_filled.back() + 1),
                    vtp_array + best_idx + 1, 0);
        }
        (*visits_to_perform.back())[best_idx] += new_visits;
        cur_limit -= new_visits;
        Node* child_node = best_edge.GetOrSpawnNode(/* parent */ node, nullptr);
        // Probably best place to check for two-fold draws consistently.
        // Depth starts with 1 at root, so real depth is depth - 1.
        EnsureNodeTwoFoldCorrectForDepth(
            child_node, current_path.size() + base_depth + 1 - 1);
        bool decremented = false;
        if (child_node->TryStartScoreUpdate()) {
          current_nstarted[best_idx]++;
          new_visits -= 1;
          decremented = true;
          if (child_node->GetN() > 0 && !child_node->IsTerminal()) {
            child_node->IncrementNInFlight(new_visits);
            current_nstarted[best_idx] += new_visits;
          }
          current_score[best_idx] = current_pol[best_idx] * puct_mult /
                                        (1 + current_nstarted[best_idx]) +
                                    current_util[best_idx];
        }
        if ((decremented &&
             (child_node->GetN() == 0 || child_node->IsTerminal()))) {
          // Reduce 1 for the visits_to_perform to ensure the collision created
          // doesn't include this visit.
          (*visits_to_perform.back())[best_idx] -= 1;
          receiver->push_back(NodeToProcess::Visit(
              child_node,
              static_cast<uint16_t>(current_path.size() + 1 + base_depth)));
          completed_visits++;
          receiver->back().moves_to_visit.reserve(moves_to_path.size() + 1);
          receiver->back().moves_to_visit = moves_to_path;
          receiver->back().moves_to_visit.push_back(best_edge.GetMove());
        }
        if (best_idx > vtp_last_filled.back() &&
            (*visits_to_perform.back())[best_idx] > 0) {
          vtp_last_filled.back() = best_idx;
        }
      }
      is_root_node = false;
      // Actively do any splits now rather than waiting for potentially long
      // tree walk to get there.
      for (int i = 0; i <= vtp_last_filled.back(); i++) {
        int child_limit = (*visits_to_perform.back())[i];
        if (params_.GetTaskWorkersPerSearchWorker() > 0 &&
            child_limit > params_.GetMinimumWorkSizeForPicking() &&
            child_limit <
                ((collision_limit - passed_off - completed_visits) * 2 / 3) &&
            child_limit + passed_off + completed_visits <
                collision_limit -
                    params_.GetMinimumRemainingWorkSizeForPicking()) {
          Node* child_node = cur_iters[i].GetOrSpawnNode(/* parent */ node);
          // Don't split if not expanded or terminal.
          if (child_node->GetN() == 0 || child_node->IsTerminal()) continue;
          bool passed = false;
          {
            // Multiple writers, so need mutex here.
            Mutex::Lock lock(picking_tasks_mutex_);
            // Ensure not to exceed size of reservation.
            if (picking_tasks_.size() < MAX_TASKS) {
              moves_to_path.push_back(cur_iters[i].GetMove());
              picking_tasks_.emplace_back(
                  child_node, current_path.size() - 1 + base_depth + 1,
                  moves_to_path, child_limit);
              moves_to_path.pop_back();
              task_count_.fetch_add(1, std::memory_order_acq_rel);
              task_added_.notify_all();
              passed = true;
              passed_off += child_limit;
            }
          }
          if (passed) {
            (*visits_to_perform.back())[i] = 0;
          }
        }
      }
      // Fall through to select the first child.
    }
    int min_idx = current_path.back();
    bool found_child = false;
    if (vtp_last_filled.back() > min_idx) {
      int idx = -1;
      for (auto& child : node->Edges()) {
        idx++;
        if (idx > min_idx && (*visits_to_perform.back())[idx] > 0) {
          if (moves_to_path.size() != current_path.size() + base_depth) {
            moves_to_path.push_back(child.GetMove());
          } else {
            moves_to_path.back() = child.GetMove();
          }
          current_path.back() = idx;
          current_path.push_back(-1);
          node = child.GetOrSpawnNode(/* parent */ node, nullptr);
          found_child = true;
          break;
        }
        if (idx >= vtp_last_filled.back()) break;
      }
    }
    if (!found_child) {
      node = node->GetParent();
      if (!moves_to_path.empty()) moves_to_path.pop_back();
      current_path.pop_back();
      vtp_buffer.push_back(std::move(visits_to_perform.back()));
      visits_to_perform.pop_back();
      vtp_last_filled.pop_back();
    }
  }
}
void SearchWorker::ExtendNode(Node* node, int depth,
                              const std::vector<Move>& moves_to_node,
                              PositionHistory* history) {
  // Initialize position sequence with pre-move position.
  history->Trim(search_->played_history_.GetLength());
  for (size_t i = 0; i < moves_to_node.size(); i++) {
    history->Append(moves_to_node[i]);
  }
  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history->Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }
  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }
    if (history->Last().GetRule50Ply() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }
    const auto repetitions = history->Last().GetRepetitions();
    // Mark two-fold repetitions as draws according to settings.
    // Depth starts with 1 at root, so number of plies in PV is depth - 1.
    if (repetitions >= 2) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    } else if (repetitions == 1 && depth - 1 >= 4 &&
               params_.GetTwoFoldDraws() &&
               depth - 1 >= history->Last().GetPliesSincePrevRepetition()) {
      const auto cycle_length = history->Last().GetPliesSincePrevRepetition();
      // use plies since first repetition as moves left; exact if forced draw.
      node->MakeTerminal(GameResult::DRAW, (float)cycle_length,
                         Node::Terminal::TwoFold);
      return;
    }
    // Neither by-position or by-rule termination, but maybe it's a TB position.
    if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
        board.castlings().no_legal_castle() &&
        history->Last().GetRule50Ply() == 0 &&
        (board.ours() | board.theirs()).count() <=
            search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history->Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // TB nodes don't have NN evaluation, assign M from parent node.
        float m = 0.0f;
        // Need a lock to access parent, in case MakeSolid is in progress.
        {
          SharedMutex::SharedLock lock(search_->nodes_mutex_);
          auto parent = node->GetParent();
          if (parent) {
            m = std::max(0.0f, parent->GetM() - 1.0f);
          }
        }
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON, m,
                             Node::Terminal::Tablebase);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON, m,
                             Node::Terminal::Tablebase);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW, m, Node::Terminal::Tablebase);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }
  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);
}
void SearchWorker::ExtendNode(Node* node, int depth) {
  std::vector<Move> to_add;
  // Could instead reserve one more than the difference between history_.size()
  // and history_.capacity().
  to_add.reserve(60);
  // Need a lock to walk parents of leaf in case MakeSolid is concurrently
  // adjusting parent chain.
  {
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    Node* cur = node;
    while (cur != search_->root_node_) {
      Node* prev = cur->GetParent();
      to_add.push_back(prev->GetEdgeToNode(cur)->GetMove());
      cur = prev;
    }
  }
  std::reverse(to_add.begin(), to_add.end());
  ExtendNode(node, depth, to_add, &history_);
}
// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node, bool add_if_cached,
                                        int* transform_out) {
  const auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation_->AddInputByHash(hash)) {
      if (transform_out) {
        *transform_out = TransformForPosition(
            search_->network_->GetCapabilities().input_format, history_);
      }
      return true;
    }
  } else {
    if (search_->cache_->ContainsKey(hash)) {
      if (transform_out) {
        *transform_out = TransformForPosition(
            search_->network_->GetCapabilities().input_format, history_);
      }
      return true;
    }
  }
  int transform;
  auto planes =
      EncodePositionForNN(search_->network_->GetCapabilities().input_format,
                          history_, 8, params_.GetHistoryFill(), &transform);
  std::vector<uint16_t> moves;
  if (node && node->HasChildren()) {
    // Legal moves are known, use them.
    moves.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) {
      moves.emplace_back(edge.GetMove().as_nn_index(transform));
    }
  } else {
    // Cache pseudolegal moves. A bit of a waste, but faster.
    const auto& pseudolegal_moves =
        history_.Last().GetBoard().GeneratePseudolegalMoves();
    moves.reserve(pseudolegal_moves.size());
    for (auto iter = pseudolegal_moves.begin(), end = pseudolegal_moves.end();
         iter != end; ++iter) {
      moves.emplace_back(iter->as_nn_index(transform));
    }
  }
  computation_->AddInput(hash, std::move(planes), std::move(moves));
  if (transform_out) *transform_out = transform;
  return false;
}
// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);
  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.node,
                                               node_to_process.multivisit);
    }
  }
}
// 3. Prefetch into cache.
// ~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::MaybePrefetchIntoCache() {
  // TODO(mooskagh) Remove prefetch into cache if node collisions work well.
  // If there are requests to NN, but the batch is not full, try to prefetch
  // nodes which are likely useful in future.
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->GetCacheMisses() > 0 &&
      computation_->GetCacheMisses() < params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(
        search_->root_node_,
        params_.GetMaxPrefetchBatch() - computation_->GetCacheMisses(), false);
  }
}
// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int SearchWorker::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth) {
  const float draw_score = search_->GetDrawScore(is_odd_depth);
  if (budget <= 0) return 0;
  // We are in a leaf, which is not yet being processed.
  if (!node || node->GetNStarted() == 0) {
    if (AddNodeToComputation(node, false, nullptr)) {
      // Make it return 0 to make it not use the slot, so that the function
      // tries hard to find something to cache even among unpopular moves.
      // In practice that slows things down a lot though, as it's not always
      // easy to find what to cache.
      return 1;
    }
    return 1;
  }
  assert(node);
  // n = 0 and n_in_flight_ > 0, that means the node is being extended.
  if (node->GetN() == 0) return 0;
  // The node is terminal; don't prefetch it.
  if (node->IsTerminal()) return 0;
  // Populate all subnodes and their scores.
  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  const float cpuct =
      ComputeCpuct(params_, node->GetN(), node == search_->root_node_);
  const float puct_mult =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu =
      GetFpu(params_, node, node == search_->root_node_, draw_score);
  for (auto& edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    // Flip the sign of a score to be able to easily sort.
    // TODO: should this use logit_q if set??
    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(fpu, draw_score),
                        edge);
  }
  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initialize for the case where there's only
                                 // one child.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;
    // Sort next chunk of a vector. 3 at a time. Most of the time it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end(),
                        [](const ScoredEdge& a, const ScoredEdge& b) {
                          return a.first < b.first;
                        });
      first_unsorted_index = new_unsorted_index;
    }
    auto edge = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, so flip it back.
      const float next_score = -scores[i + 1].first;
      // TODO: As above - should this use logit_q if set?
      const float q = edge.GetQ(-fpu, draw_score);
      if (next_score > q) {
        budget_to_spend =
            std::min(budget, int(edge.GetP() * puct_mult / (next_score - q) -
                                 edge.GetNStarted()) +
                                 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history_.Append(edge.GetMove());
    const int budget_spent =
        PrefetchIntoCache(edge.node(), budget_to_spend, !is_odd_depth);
    history_.Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}
// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() { computation_->ComputeBlocking(); }
// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  // Populate NN/cached results, or terminal results, into nodes.
  int idx_in_computation = 0;
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process, *computation_, idx_in_computation);
    if (node_to_process.nn_queried) ++idx_in_computation;
  }
}
template <typename Computation>
void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process,
                                         const Computation& computation,
                                         int idx_in_computation) {
  if (node_to_process->IsCollision()) return;
  Node* node = node_to_process->node;
  if (!node_to_process->nn_queried) {
    // Terminal nodes don't involve the neural NetworkComputation, nor do
    // they require any further processing after value retrieval.
    node_to_process->v = node->GetWL();
    node_to_process->d = node->GetD();
    node_to_process->m = node->GetM();
    return;
  }
  // For NN results, we need to populate policy as well as value.
  // First the value...
  node_to_process->v = -computation.GetQVal(idx_in_computation);
  node_to_process->d = computation.GetDVal(idx_in_computation);
  node_to_process->m = computation.GetMVal(idx_in_computation);
  // ...and secondly, the policy data.
  // Calculate maximum first.
  float max_p = -std::numeric_limits<float>::infinity();
  // Intermediate array to store values when processing policy.
  // There are never more than 256 valid legal moves in any legal position.
  std::array<float, 256> intermediate;
  int counter = 0;
  for (auto& edge : node->Edges()) {
    float p = computation.GetPVal(
        idx_in_computation,
        edge.GetMove().as_nn_index(node_to_process->probability_transform));
    intermediate[counter++] = p;
    max_p = std::max(max_p, p);
  }
  float total = 0.0;
  for (int i = 0; i < counter; i++) {
    // Perform softmax and take into account policy softmax temperature T.
    // Note that we want to calculate (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
    float p =
        FastExp((intermediate[i] - max_p) / params_.GetPolicySoftmaxTemp());
    intermediate[i] = p;
    total += p;
  }
  counter = 0;
  // Normalize P values to add up to 1.0.
  const float scale = total > 0.0f ? 1.0f / total : 1.0f;
  for (auto& edge : node->Edges()) {
    edge.edge()->SetP(intermediate[counter++] * scale);
  }
  // Add Dirichlet noise if enabled and at root.
  if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
    ApplyDirichletNoise(node, params_.GetNoiseEpsilon(),
                        params_.GetNoiseAlpha());
  }
  node->SortEdges();
}
// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  // Nodes mutex for doing node updates.
  SharedMutex::Lock lock(search_->nodes_mutex_);
  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}
void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (node_to_process.IsCollision()) {
    // Collisions are handled via shared_collisions instead.
    return;
  }
  // For the first visit to a terminal, maybe update parent bounds too.
  auto update_parent_bounds =
      params_.GetStickyEndgames() && node->IsTerminal() && !node->GetN();
  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.v;
  float d = node_to_process.d;
  float m = node_to_process.m;
  int n_to_fix = 0;
  float v_delta = 0.0f;
  float d_delta = 0.0f;
  float m_delta = 0.0f;
  uint32_t solid_threshold =
      static_cast<uint32_t>(params_.GetSolidTreeThreshold());
  for (Node *n = node, *p; n != search_->root_node_->GetParent(); n = p) {
    p = n->GetParent();
    // Current node might have become terminal from some other descendant, so
    // backup the rest of the way with more accurate values.
    if (n->IsTerminal()) {
      v = n->GetWL();
      d = n->GetD();
      m = n->GetM();
    }
    n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    if (n_to_fix > 0 && !n->IsTerminal()) {
      n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }
    if (n->GetN() >= solid_threshold) {
      if (n->MakeSolid() && n == search_->root_node_) {
        // If we make the root solid, the current_best_edge_ becomes invalid and
        // we should repopulate it.
        search_->current_best_edge_ =
            search_->GetBestChildNoTemperature(search_->root_node_, 0);
      }
    }
    // Nothing left to do without ancestors to update.
    if (!p) break;
    bool old_update_parent_bounds = update_parent_bounds;
    // If parent already is terminal further adjustment is not required.
    if (p->IsTerminal()) n_to_fix = 0;
    // Try setting parent bounds except the root or those already terminal.
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !p->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);
    // Q will be flipped for opponent.
    v = -v;
    v_delta = -v_delta;
    m++;
    // Update the stats.
    // Best move.
    // If update_parent_bounds was set, we just adjusted bounds on the
    // previous loop or there was no previous loop, so if n is a terminal, it
    // just became that way and could be a candidate for changing the current
    // best edge. Otherwise a visit can only change best edge if its to an edge
    // that isn't already the best and the new n is equal or greater to the old
    // n.
    if (p == search_->root_node_ &&
        ((old_update_parent_bounds && n->IsTerminal()) ||
         (n != search_->current_best_edge_.node() &&
          search_->current_best_edge_.GetN() <= n->GetN()))) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_, 0);
    }
  }
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}
bool SearchWorker::MaybeSetBounds(Node* p, float m, int* n_to_fix,
                                  float* v_delta, float* d_delta,
                                  float* m_delta) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;
  // Determine the maximum (lower, upper) bounds across all children.
  // (-1,-1) Loss (initial and lowest bounds)
  // (-1, 0) Can't Win
  // (-1, 1) Regular node
  // ( 0, 0) Draw
  // ( 0, 1) Can't Lose
  // ( 1, 1) Win (highest bounds)
  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;
  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);
    // Checkmate is the best, so short-circuit.
    const auto is_tb = edge.IsTbTerminal();
    if (edge_lower == GameResult::WHITE_WON && !is_tb) {
      prefer_tb = false;
      break;
    } else if (edge_upper == GameResult::BLACK_WON) {
      // Track the longest loss.
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
    prefer_tb = prefer_tb || is_tb;
  }
  // The parent's bounds are flipped from the children (-max(U), -max(L))
  // aggregated as if it was a single child (forced move) of the same bound.
  //       Loss (-1,-1) -> ( 1, 1) Win
  //  Can't Win (-1, 0) -> ( 0, 1) Can't Lose
  //    Regular (-1, 1) -> (-1, 1) Regular
  //       Draw ( 0, 0) -> ( 0, 0) Draw
  // Can't Lose ( 0, 1) -> (-1, 0) Can't Win
  //        Win ( 1, 1) -> (-1,-1) Loss
  // Nothing left to do for ancestors if the parent would be a regular node.
  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    // Search can stop at the parent if the bounds can't change anymore, so make
    // it terminal preferring shorter wins and longer losses.
    *n_to_fix = p->GetN();
    assert(*n_to_fix > 0);
    float cur_v = p->GetWL();
    float cur_d = p->GetD();
    float cur_m = p->GetM();
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Node::Terminal::Tablebase : Node::Terminal::EndOfGame);
    // Negate v_delta because we're calculating for the parent, but immediately
    // afterwards we'll negate v_delta in case it has come from the child.
    *v_delta = -(p->GetWL() - cur_v);
    *d_delta = p->GetD() - cur_d;
    *m_delta = p->GetM() - cur_m;
  } else {
    p->SetBounds(-upper, -lower);
  }
  // Bounds were set, so indicate we should check the parent too.
  return true;
}
// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();
  // If this thread had no work, not even out of order, then sleep for some
  // milliseconds. Collisions don't count as work, so have to enumerate to find
  // out if there was anything done.
  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/search.cc

// begin of /Users/syys/CLionProjects/lc0/src/neural/loader.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#ifdef _WIN32
#else
#endif
namespace lczero {
namespace {
const std::uint32_t kWeightMagic = 0x1c0;
std::string DecompressGzip(const std::string& filename) {
  const int kStartingSize = 8 * 1024 * 1024;  // 8M
  std::string buffer;
  buffer.resize(kStartingSize);
  int bytes_read = 0;
  // Read whole file into a buffer.
  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw Exception("Cannot read weights from " + filename);
  }
  if (filename == CommandLine::BinaryName()) {
    // The network file should be appended at the end of the lc0 executable,
    // followed by the network file size and a "Lc0!" (0x2130634c) magic.
    int32_t size, magic;
    if (fseek(fp, -8, SEEK_END) || fread(&size, 4, 1, fp) != 1 ||
        fread(&magic, 4, 1, fp) != 1 || magic != 0x2130634c) {
      fclose(fp);
      throw Exception("No embedded file detected.");
    }
    fseek(fp, -size - 8, SEEK_END);
  }
  fflush(fp);
  gzFile file = gzdopen(dup(fileno(fp)), "rb");
  fclose(fp);
  if (!file) {
    throw Exception("Cannot process file " + filename);
  }
  while (true) {
    const int sz =
        gzread(file, &buffer[bytes_read], buffer.size() - bytes_read);
    if (sz < 0) {
      int errnum;
      throw Exception(gzerror(file, &errnum));
    }
    if (sz == static_cast<int>(buffer.size()) - bytes_read) {
      bytes_read = buffer.size();
      buffer.resize(buffer.size() * 2);
    } else {
      bytes_read += sz;
      buffer.resize(bytes_read);
      break;
    }
  }
  gzclose(file);
  return buffer;
}
void FixOlderWeightsFile(WeightsFile* file) {
  using nf = pblczero::NetworkFormat;
  auto network_format = file->format().network_format().network();
  const auto has_network_format = file->format().has_network_format();
  if (has_network_format && network_format != nf::NETWORK_CLASSICAL &&
      network_format != nf::NETWORK_SE) {
    // Already in a new format, return unchanged.
    return;
  }
  auto* net = file->mutable_format()->mutable_network_format();
  if (!has_network_format) {
    // Older protobufs don't have format definition.
    net->set_input(nf::INPUT_CLASSICAL_112_PLANE);
    net->set_output(nf::OUTPUT_CLASSICAL);
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == pblczero::NetworkFormat::NETWORK_CLASSICAL) {
    // Populate policyFormat and valueFormat fields in old protobufs
    // without these fields.
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == pblczero::NetworkFormat::NETWORK_SE) {
    net->set_network(nf::NETWORK_SE_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  }
}
WeightsFile ParseWeightsProto(const std::string& buffer) {
  WeightsFile net;
  net.ParseFromString(buffer);
  if (net.magic() != kWeightMagic) {
    throw Exception("Invalid weight file: bad header.");
  }
  const auto min_version =
      GetVersionStr(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch(), "", "");
  const auto lc0_ver = GetVersionInt();
  const auto net_ver =
      GetVersionInt(net.min_version().major(), net.min_version().minor(),
                    net.min_version().patch());
  FixOlderWeightsFile(&net);
  // Weights files with this signature are also compatible.
  if (net_ver != 0x5c99973 && net_ver > lc0_ver) {
    throw Exception("Invalid weight file: lc0 version >= " + min_version +
                    " required.");
  }
  if (net.has_weights() &&
      net.format().weights_encoding() != pblczero::Format::LINEAR16) {
    throw Exception("Invalid weight file: unsupported encoding.");
  }
  return net;
}
}  // namespace
WeightsFile LoadWeightsFromFile(const std::string& filename) {
  FloatVectors vecs;
  auto buffer = DecompressGzip(filename);
  if (buffer.size() < 2) {
    throw Exception("Invalid weight file: too small.");
  }
  if (buffer[0] == '1' && buffer[1] == '\n') {
    throw Exception("Invalid weight file: no longer supported.");
  }
  if (buffer[0] == '2' && buffer[1] == '\n') {
    throw Exception(
        "Text format weights files are no longer supported. Use a command line "
        "tool to convert it to the new format.");
  }
  return ParseWeightsProto(buffer);
}
std::string DiscoverWeightsFile() {
  const int kMinFileSize = 500000;  // 500 KB
  std::vector<std::string> data_dirs = {CommandLine::BinaryDirectory()};
  const std::string user_data_path = GetUserDataDirectory();
  if (!user_data_path.empty()) {
    data_dirs.emplace_back(user_data_path + "lc0");
  }
  for (const auto& dir : GetSystemDataDirectoryList()) {
    data_dirs.emplace_back(dir + (dir.back() == '/' ? "" : "/") + "lc0");
  }
  for (const auto& dir : data_dirs) {
    // Open all files in <dir> amd <dir>/networks,
    // ones which are >= kMinFileSize are candidates.
    std::vector<std::pair<time_t, std::string> > time_and_filename;
    for (const auto& path : {"", "/networks"}) {
      for (const auto& file : GetFileList(dir + path)) {
        const std::string filename = dir + path + "/" + file;
        if (GetFileSize(filename) < kMinFileSize) continue;
        time_and_filename.emplace_back(GetFileTime(filename), filename);
      }
    }
    std::sort(time_and_filename.rbegin(), time_and_filename.rend());
    // Open all candidates, from newest to oldest, possibly gzipped, and try to
    // read version for it. If version is 2 or if the file is our protobuf,
    // return it.
    for (const auto& candidate : time_and_filename) {
      const gzFile file = gzopen(candidate.second.c_str(), "rb");
      if (!file) continue;
      unsigned char buf[256];
      int sz = gzread(file, buf, 256);
      gzclose(file);
      if (sz < 0) continue;
      std::string str(buf, buf + sz);
      std::istringstream data(str);
      int val = 0;
      data >> val;
      if (!data.fail() && val == 2) {
        CERR << "Found txt network file: " << candidate.second;
        return candidate.second;
      }
      // First byte of the protobuf stream is 0x0d for fixed32, so we ignore it
      // as our own magic should suffice.
      const auto magic = buf[1] | (static_cast<uint32_t>(buf[2]) << 8) |
                         (static_cast<uint32_t>(buf[3]) << 16) |
                         (static_cast<uint32_t>(buf[4]) << 24);
      if (magic == kWeightMagic) {
        CERR << "Found pb network file: " << candidate.second;
        return candidate.second;
      }
    }
  }
  LOGFILE << "Network weights file not found.";
  return {};
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/loader.cc

// begin of /Users/syys/CLionProjects/lc0/src/neural/factory.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
const OptionId NetworkFactory::kWeightsId{
    "weights", "WeightsFile",
    "Path from which to load network weights.\nSetting it to <autodiscover> "
    "makes it search in ./ and ./weights/ subdirectories for the latest (by "
    "file date) file which looks like weights.",
    'w'};
const OptionId NetworkFactory::kBackendId{
    "backend", "Backend", "Neural network computational backend to use.", 'b'};
const OptionId NetworkFactory::kBackendOptionsId{
    "backend-opts", "BackendOptions",
    "Parameters of neural network backend. "
    "Exact parameters differ per backend.",
    'o'};
const char* kAutoDiscover = "<autodiscover>";
const char* kEmbed = "<built in>";
NetworkFactory* NetworkFactory::Get() {
  static NetworkFactory factory;
  return &factory;
}
NetworkFactory::Register::Register(const std::string& name, FactoryFunc factory,
                                   int priority) {
  NetworkFactory::Get()->RegisterNetwork(name, factory, priority);
}
void NetworkFactory::PopulateOptions(OptionsParser* options) {
#if defined(EMBED)
  options->Add<StringOption>(NetworkFactory::kWeightsId) = kEmbed;
#else
  options->Add<StringOption>(NetworkFactory::kWeightsId) = kAutoDiscover;
#endif
  const auto backends = NetworkFactory::Get()->GetBackendsList();
  options->Add<ChoiceOption>(NetworkFactory::kBackendId, backends) =
      backends.empty() ? "<none>" : backends[0];
  options->Add<StringOption>(NetworkFactory::kBackendOptionsId);
}
void NetworkFactory::RegisterNetwork(const std::string& name,
                                     FactoryFunc factory, int priority) {
  factories_.emplace_back(name, factory, priority);
  std::sort(factories_.begin(), factories_.end());
}
std::vector<std::string> NetworkFactory::GetBackendsList() const {
  std::vector<std::string> result;
  for (const auto& x : factories_) result.emplace_back(x.name);
  return result;
}
std::unique_ptr<Network> NetworkFactory::Create(
    const std::string& network, const std::optional<WeightsFile>& weights,
    const OptionsDict& options) {
  CERR << "Creating backend [" << network << "]...";
  for (const auto& factory : factories_) {
    if (factory.name == network) {
      return factory.factory(weights, options);
    }
  }
  throw Exception("Unknown backend: " + network);
}
NetworkFactory::BackendConfiguration::BackendConfiguration(
    const OptionsDict& options)
    : weights_path(options.Get<std::string>(kWeightsId)),
      backend(options.Get<std::string>(kBackendId)),
      backend_options(options.Get<std::string>(kBackendOptionsId)) {}
bool NetworkFactory::BackendConfiguration::operator==(
    const BackendConfiguration& other) const {
  return (weights_path == other.weights_path && backend == other.backend &&
          backend_options == other.backend_options);
}
std::unique_ptr<Network> NetworkFactory::LoadNetwork(
    const OptionsDict& options) {
  std::string net_path = options.Get<std::string>(kWeightsId);
  const std::string backend = options.Get<std::string>(kBackendId);
  const std::string backend_options =
      options.Get<std::string>(kBackendOptionsId);
  if (net_path == kAutoDiscover) {
    net_path = DiscoverWeightsFile();
  } else if (net_path == kEmbed) {
    net_path = CommandLine::BinaryName();
  } else {
    CERR << "Loading weights file from: " << net_path;
  }
  std::optional<WeightsFile> weights;
  if (!net_path.empty()) {
    weights = LoadWeightsFromFile(net_path);
  }
  OptionsDict network_options(&options);
  network_options.AddSubdictFromString(backend_options);
  auto ptr = NetworkFactory::Get()->Create(backend, weights, network_options);
  network_options.CheckAllOptionsRead(backend);
  return ptr;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/neural/factory.cc

// begin of /Users/syys/CLionProjects/lc0/src/benchmark/benchmark.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const int kDefaultThreads = 2;
const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kNodesId{"nodes", "", "Number of nodes to run as a benchmark."};
const OptionId kMovetimeId{"movetime", "",
                           "Benchmark time allocation, in milliseconds."};
const OptionId kFenId{"fen", "", "Benchmark position FEN."};
const OptionId kNumPositionsId{"num-positions", "",
                               "The number of benchmark positions to test."};
}  // namespace
void Benchmark::Run() {
  OptionsParser options;
  NetworkFactory::PopulateOptions(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options.Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  SearchParams::Populate(&options);
  options.Add<IntOption>(kNodesId, -1, 999999999) = -1;
  options.Add<IntOption>(kMovetimeId, -1, 999999999) = 10000;
  options.Add<StringOption>(kFenId) = "";
  options.Add<IntOption>(kNumPositionsId, 1, 34) = 34;
  if (!options.ProcessAllFlags()) return;
  try {
    auto option_dict = options.GetOptionsDict();
    auto network = NetworkFactory::LoadNetwork(option_dict);
    const int visits = option_dict.Get<int>(kNodesId);
    const int movetime = option_dict.Get<int>(kMovetimeId);
    const std::string fen = option_dict.Get<std::string>(kFenId);
    int num_positions = option_dict.Get<int>(kNumPositionsId);
    std::vector<std::double_t> times;
    std::vector<std::int64_t> playouts;
    std::uint64_t cnt = 1;
    if (fen.length() > 0) {
      positions = {fen};
      num_positions = 1;
    }
    std::vector<std::string> testing_positions(
        positions.cbegin(), positions.cbegin() + num_positions);
    for (std::string position : testing_positions) {
      std::cout << "\nPosition: " << cnt++ << "/" << testing_positions.size()
                << " " << position << std::endl;
      auto stopper = std::make_unique<ChainedSearchStopper>();
      if (movetime > -1) {
        stopper->AddStopper(std::make_unique<TimeLimitStopper>(movetime));
      }
      if (visits > -1) {
        stopper->AddStopper(std::make_unique<VisitsStopper>(visits, false));
      }
      NNCache cache;
      cache.SetCapacity(option_dict.Get<int>(kNNCacheSizeId));
      NodeTree tree;
      tree.ResetToPosition(position, {});
      const auto start = std::chrono::steady_clock::now();
      auto search = std::make_unique<Search>(
          tree, network.get(),
          std::make_unique<CallbackUciResponder>(
              std::bind(&Benchmark::OnBestMove, this, std::placeholders::_1),
              std::bind(&Benchmark::OnInfo, this, std::placeholders::_1)),
          MoveList(), start, std::move(stopper), false, option_dict, &cache,
          nullptr);
      search->StartThreads(option_dict.Get<int>(kThreadsOptionId));
      search->Wait();
      const auto end = std::chrono::steady_clock::now();
      const auto time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      times.push_back(time.count());
      playouts.push_back(search->GetTotalPlayouts());
    }
    const auto total_playouts =
        std::accumulate(playouts.begin(), playouts.end(), 0);
    const auto total_time = std::accumulate(times.begin(), times.end(), 0);
    std::cout << "\n==========================="
              << "\nTotal time (ms) : " << total_time
              << "\nNodes searched  : " << total_playouts
              << "\nNodes/second    : "
              << std::lround(1000.0 * total_playouts / (total_time + 1))
              << std::endl;
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}
void Benchmark::OnBestMove(const BestMoveInfo& move) {
  std::cout << "bestmove " << move.bestmove.as_string() << std::endl;
}
void Benchmark::OnInfo(const std::vector<ThinkingInfo>& infos) {
  std::string line = "Benchmark time " + std::to_string(infos[0].time);
  line += " ms, " + std::to_string(infos[0].nodes) + " nodes, ";
  line += std::to_string(infos[0].nps) + " nps";
  if (!infos[0].pv.empty()) line += ", move " + infos[0].pv[0].as_string();
  std::cout << line << std::endl;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/benchmark/benchmark.cc

// begin of /Users/syys/CLionProjects/lc0/src/engine.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const int kDefaultThreads = 2;
const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
const OptionId kPonderId{"ponder", "Ponder",
                         "This option is ignored. Here to please chess GUIs."};
const OptionId kUciChess960{
    "chess960", "UCI_Chess960",
    "Castling moves are encoded as \"king takes rook\"."};
const OptionId kShowWDL{"show-wdl", "UCI_ShowWDL",
                        "Show win, draw and lose probability."};
const OptionId kShowMovesleft{"show-movesleft", "UCI_ShowMovesLeft",
                              "Show estimated moves left."};
const OptionId kStrictUciTiming{"strict-uci-timing", "StrictTiming",
                                "The UCI host compensates for lag, waits for "
                                "the 'readyok' reply before sending 'go' and "
                                "only then starts timing."};
const OptionId kPreload{"preload", "",
                        "Initialize backend and load net on engine startup."};
MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList result;
  if (moves.size()) {
    result.reserve(moves.size());
    const auto legal_moves = board.GenerateLegalMoves();
    const auto end = legal_moves.end();
    for (const auto& move : moves) {
      const auto m = board.GetModernMove({move, board.flipped()});
      if (std::find(legal_moves.begin(), end, m) != end) result.emplace_back(m);
    }
    if (result.empty()) throw Exception("No legal searchmoves.");
  }
  return result;
}
}  // namespace
EngineController::EngineController(std::unique_ptr<UciResponder> uci_responder,
                                   const OptionsDict& options)
    : options_(options),
      uci_responder_(std::move(uci_responder)),
      current_position_{ChessBoard::kStartposFen, {}} {}
void EngineController::PopulateOptions(OptionsParser* options) {
  using namespace std::placeholders;
  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  SearchParams::Populate(options);
  options->Add<StringOption>(kSyzygyTablebaseId);
  // Add "Ponder" option to signal to GUIs that we support pondering.
  // This option is currently not used by lc0 in any way.
  options->Add<BoolOption>(kPonderId) = true;
  options->Add<BoolOption>(kUciChess960) = false;
  options->Add<BoolOption>(kShowWDL) = false;
  options->Add<BoolOption>(kShowMovesleft) = false;
  ConfigFile::PopulateOptions(options);
  PopulateTimeManagementOptions(RunType::kUci, options);
  options->Add<BoolOption>(kStrictUciTiming) = false;
  options->HideOption(kStrictUciTiming);
  options->Add<BoolOption>(kPreload) = false;
}
void EngineController::ResetMoveTimer() {
  move_start_time_ = std::chrono::steady_clock::now();
}
// Updates values from Uci options.
void EngineController::UpdateFromUciOptions() {
  SharedLock lock(busy_mutex_);
  // Syzygy tablebases.
  std::string tb_paths = options_.Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty() && tb_paths != tb_paths_) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    } else {
      tb_paths_ = tb_paths;
    }
  }
  // Network.
  const auto network_configuration =
      NetworkFactory::BackendConfiguration(options_);
  if (network_configuration_ != network_configuration) {
    network_ = NetworkFactory::LoadNetwork(options_);
    network_configuration_ = network_configuration;
  }
  // Cache size.
  cache_.SetCapacity(options_.Get<int>(kNNCacheSizeId));
  // Check whether we can update the move timer in "Go".
  strict_uci_timing_ = options_.Get<bool>(kStrictUciTiming);
}
void EngineController::EnsureReady() {
  std::unique_lock<RpSharedMutex> lock(busy_mutex_);
  // If a UCI host is waiting for our ready response, we can consider the move
  // not started until we're done ensuring ready.
  ResetMoveTimer();
}
void EngineController::NewGame() {
  // In case anything relies upon defaulting to default position and just calls
  // newgame and goes straight into go.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  cache_.Clear();
  search_.reset();
  tree_.reset();
  CreateFreshTimeManager();
  current_position_ = {ChessBoard::kStartposFen, {}};
  UpdateFromUciOptions();
}
void EngineController::SetPosition(const std::string& fen,
                                   const std::vector<std::string>& moves_str) {
  // Some UCI hosts just call position then immediately call go, while starting
  // the clock on calling 'position'.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  current_position_ = CurrentPosition{fen, moves_str};
  search_.reset();
}
Position EngineController::ApplyPositionMoves() {
  ChessBoard board;
  int no_capture_ply;
  int game_move;
  board.SetFromFen(current_position_.fen, &no_capture_ply, &game_move);
  int game_ply = 2 * game_move - (board.flipped() ? 1 : 2);
  Position pos(board, no_capture_ply, game_ply);
  for (std::string move_str: current_position_.moves) {
    Move move(move_str);
    if (pos.IsBlackToMove()) move.Mirror();
    pos = Position(pos, move);
  }
  return pos;
}
void EngineController::SetupPosition(
    const std::string& fen, const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();
  UpdateFromUciOptions();
  if (!tree_) tree_ = std::make_unique<NodeTree>();
  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  const bool is_same_game = tree_->ResetToPosition(fen, moves);
  if (!is_same_game) CreateFreshTimeManager();
}
void EngineController::CreateFreshTimeManager() {
  time_manager_ = MakeTimeManager(options_);
}
namespace {
class PonderResponseTransformer : public TransformingUciResponder {
 public:
  PonderResponseTransformer(std::unique_ptr<UciResponder> parent,
                            std::string ponder_move)
      : TransformingUciResponder(std::move(parent)),
        ponder_move_(std::move(ponder_move)) {}
  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    // Output all stats from main variation (not necessary the ponder move)
    // but PV only from ponder move.
    ThinkingInfo ponder_info;
    for (const auto& info : *infos) {
      if (info.multipv <= 1) {
        ponder_info = info;
        if (ponder_info.mate) ponder_info.mate = -*ponder_info.mate;
        if (ponder_info.score) ponder_info.score = -*ponder_info.score;
        if (ponder_info.depth > 1) ponder_info.depth--;
        if (ponder_info.seldepth > 1) ponder_info.seldepth--;
        ponder_info.pv.clear();
      }
      if (!info.pv.empty() && info.pv[0].as_string() == ponder_move_) {
        ponder_info.pv.assign(info.pv.begin() + 1, info.pv.end());
      }
    }
    infos->clear();
    infos->push_back(ponder_info);
  }
 private:
  std::string ponder_move_;
};
}  // namespace
void EngineController::Go(const GoParams& params) {
  // TODO: should consecutive calls to go be considered to be a continuation and
  // hence have the same start time like this behaves, or should we check start
  // time hasn't changed since last call to go and capture the new start time
  // now?
  if (strict_uci_timing_ || !move_start_time_) ResetMoveTimer();
  go_params_ = params;
  std::unique_ptr<UciResponder> responder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_.get());
  // Setting up current position, now that it's known whether it's ponder or
  // not.
  if (params.ponder && !current_position_.moves.empty()) {
    std::vector<std::string> moves(current_position_.moves);
    std::string ponder_move = moves.back();
    moves.pop_back();
    SetupPosition(current_position_.fen, moves);
    responder = std::make_unique<PonderResponseTransformer>(
        std::move(responder), ponder_move);
  } else {
    SetupPosition(current_position_.fen, current_position_.moves);
  }
  if (!options_.Get<bool>(kUciChess960)) {
    // Remap FRC castling to legacy castling.
    responder = std::make_unique<Chess960Transformer>(
        std::move(responder), tree_->HeadPosition().GetBoard());
  }
  if (!options_.Get<bool>(kShowWDL)) {
    // Strip WDL information from the response.
    responder = std::make_unique<WDLResponseFilter>(std::move(responder));
  }
  if (!options_.Get<bool>(kShowMovesleft)) {
    // Strip movesleft information from the response.
    responder = std::make_unique<MovesLeftResponseFilter>(std::move(responder));
  }
  auto stopper = time_manager_->GetStopper(params, *tree_.get());
  search_ = std::make_unique<Search>(
      *tree_, network_.get(), std::move(responder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite || params.ponder,
      options_, &cache_, syzygy_tb_.get());
  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_.Get<int>(kThreadsOptionId));
}
void EngineController::PonderHit() {
  ResetMoveTimer();
  go_params_.ponder = false;
  Go(go_params_);
}
void EngineController::Stop() {
  if (search_) search_->Stop();
}
EngineLoop::EngineLoop()
    : engine_(
          std::make_unique<CallbackUciResponder>(
              std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
              std::bind(&UciLoop::SendInfo, this, std::placeholders::_1)),
          options_.GetOptionsDict()) {
  engine_.PopulateOptions(&options_);
  options_.Add<StringOption>(kLogFileId);
}
void EngineLoop::RunLoop() {
  if (!ConfigFile::Init() || !options_.ProcessAllFlags()) return;
  const auto options = options_.GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
  if (options.Get<bool>(kPreload)) engine_.NewGame();
  UciLoop::RunLoop();
}
void EngineLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}
void EngineLoop::CmdIsReady() {
  engine_.EnsureReady();
  SendResponse("readyok");
}
void EngineLoop::CmdSetOption(const std::string& name, const std::string& value,
                              const std::string& context) {
  options_.SetUciOption(name, value, context);
  // Set the log filename for the case it was set in UCI option.
  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));
}
void EngineLoop::CmdUciNewGame() { engine_.NewGame(); }
void EngineLoop::CmdPosition(const std::string& position,
                             const std::vector<std::string>& moves) {
  std::string fen = position;
  if (fen.empty()) {
    fen = ChessBoard::kStartposFen;
  }
  engine_.SetPosition(fen, moves);
}
void EngineLoop::CmdFen() {
  std::string fen = GetFen(engine_.ApplyPositionMoves());
  return SendResponse(fen);
}
void EngineLoop::CmdGo(const GoParams& params) { engine_.Go(params); }
void EngineLoop::CmdPonderHit() { engine_.PonderHit(); }
void EngineLoop::CmdStop() { engine_.Stop(); }
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/engine.cc

// begin of /Users/syys/CLionProjects/lc0/src/lc0ctl/describenet.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const OptionId kWeightsFilenameId{"weights", "WeightsFile",
                                  "Path of the input Lc0 weights file."};
bool ProcessParameters(OptionsParser* options) {
  options->Add<StringOption>(kWeightsFilenameId);
  if (!options->ProcessAllFlags()) return false;
  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kWeightsFilenameId);
  return true;
}
std::string Justify(std::string str, size_t length = 30) {
  if (str.size() + 2 < length) {
    str = std::string(length - 2 - str.size(), ' ') + str;
  }
  str += ": ";
  return str;
}
}  // namespace
void ShowNetworkGenericInfo(const pblczero::Net& weights) {
  const auto& version = weights.min_version();
  COUT << "\nGeneral";
  COUT << "~~~~~~~";
  COUT << Justify("Minimal Lc0 version") << "v" << version.major() << '.'
       << version.minor() << '.' << version.patch();
}
void ShowNetworkFormatInfo(const pblczero::Net& weights) {
  const auto& format = weights.format();
  const auto& net_format = format.network_format();
  using pblczero::Format;
  using pblczero::NetworkFormat;
  COUT << "\nFormat";
  COUT << "~~~~~~";
  if (format.has_weights_encoding()) {
    COUT << Justify("Weights encoding")
         << Format::Encoding_Name(format.weights_encoding());
  }
  if (net_format.has_input()) {
    COUT << Justify("Input")
         << NetworkFormat::InputFormat_Name(net_format.input());
  }
  if (net_format.has_network()) {
    COUT << Justify("Network")
         << NetworkFormat::NetworkStructure_Name(net_format.network());
  }
  if (net_format.has_policy()) {
    COUT << Justify("Policy")
         << NetworkFormat::PolicyFormat_Name(net_format.policy());
  }
  if (net_format.has_value()) {
    COUT << Justify("Value")
         << NetworkFormat::ValueFormat_Name(net_format.value());
  }
  if (net_format.has_moves_left()) {
    COUT << Justify("MLH")
         << NetworkFormat::MovesLeftFormat_Name(net_format.moves_left());
  }
}
void ShowNetworkTrainingInfo(const pblczero::Net& weights) {
  if (!weights.has_training_params()) return;
  COUT << "\nTraining Parameters";
  COUT << "~~~~~~~~~~~~~~~~~~~";
  using pblczero::TrainingParams;
  const auto& params = weights.training_params();
  if (params.has_training_steps()) {
    COUT << Justify("Training steps") << params.training_steps();
  }
  if (params.has_learning_rate()) {
    COUT << Justify("Learning rate") << params.learning_rate();
  }
  if (params.has_mse_loss()) {
    COUT << Justify("MSE loss") << params.mse_loss();
  }
  if (params.has_policy_loss()) {
    COUT << Justify("Policy loss") << params.policy_loss();
  }
  if (params.has_accuracy()) {
    COUT << Justify("Accuracy") << params.accuracy();
  }
  if (params.has_lc0_params()) {
    COUT << Justify("Lc0 Params") << params.lc0_params();
  }
}
void ShowNetworkWeightsInfo(const pblczero::Net& weights) {
  if (!weights.has_weights()) return;
  COUT << "\nWeights";
  COUT << "~~~~~~~";
  const auto& w = weights.weights();
  COUT << Justify("Blocks") << w.residual_size();
  COUT << Justify("Filters")
       << w.input().weights().params().size() / 2 / 112 / 9;
  COUT << Justify("Policy") << (w.has_policy1() ? "Convolution" : "Dense");
  COUT << Justify("Value")
       << (w.ip2_val_w().params().size() / 2 % 3 == 0 ? "WDL" : "Classical");
  COUT << Justify("MLH") << (w.has_moves_left() ? "Present" : "Absent");
}
void ShowNetworkOnnxInfo(const pblczero::Net& weights,
                         bool show_onnx_internals) {
  if (!weights.has_onnx_model()) return;
  const auto& onnx_model = weights.onnx_model();
  COUT << "\nONNX interface";
  COUT << "~~~~~~~~~~~~~~";
  if (onnx_model.has_data_type()) {
    COUT << Justify("Data type")
         << pblczero::OnnxModel::DataType_Name(onnx_model.data_type());
  }
  if (onnx_model.has_input_planes()) {
    COUT << Justify("Input planes") << onnx_model.input_planes();
  }
  if (onnx_model.has_output_value()) {
    COUT << Justify("Output value") << onnx_model.output_value();
  }
  if (onnx_model.has_output_wdl()) {
    COUT << Justify("Output WDL") << onnx_model.output_wdl();
  }
  if (onnx_model.has_output_policy()) {
    COUT << Justify("Output Policy") << onnx_model.output_policy();
  }
  if (onnx_model.has_output_mlh()) {
    COUT << Justify("Output MLH") << onnx_model.output_mlh();
  }
  if (!show_onnx_internals) return;
  if (!onnx_model.has_model()) return;
  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_model.model());
  COUT << "\nONNX model";
  COUT << "~~~~~~~~~~";
  if (onnx.has_ir_version()) {
    COUT << Justify("IR version") << onnx.ir_version();
  }
  if (onnx.has_producer_name()) {
    COUT << Justify("Producer Name") << onnx.producer_name();
  }
  if (onnx.has_producer_version()) {
    COUT << Justify("Producer Version") << onnx.producer_version();
  }
  if (onnx.has_domain()) {
    COUT << Justify("Domain") << onnx.domain();
  }
  if (onnx.has_model_version()) {
    COUT << Justify("Model Version") << onnx.model_version();
  }
  if (onnx.has_doc_string()) {
    COUT << Justify("Doc String") << onnx.doc_string();
  }
  for (const auto& input : onnx.graph().input()) {
    std::string name(input.name());
    if (input.has_doc_string()) {
      name += " (" + std::string(input.doc_string()) + ")";
    }
    COUT << Justify("Input") << name;
  }
  for (const auto& output : onnx.graph().output()) {
    std::string name(output.name());
    if (output.has_doc_string()) {
      name += " (" + std::string(output.doc_string()) + ")";
    }
    COUT << Justify("Output") << name;
  }
  for (const auto& opset : onnx.opset_import()) {
    std::string name;
    if (opset.has_domain()) name += std::string(opset.domain()) + " ";
    COUT << Justify("Opset") << name << opset.version();
  }
}
void ShowAllNetworkInfo(const pblczero::Net& weights) {
  ShowNetworkGenericInfo(weights);
  ShowNetworkFormatInfo(weights);
  ShowNetworkTrainingInfo(weights);
  ShowNetworkWeightsInfo(weights);
  ShowNetworkOnnxInfo(weights, true);
}
void DescribeNetworkCmd() {
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;
  const OptionsDict& dict = options_parser.GetOptionsDict();
  auto weights_file =
      LoadWeightsFromFile(dict.Get<std::string>(kWeightsFilenameId));
  ShowAllNetworkInfo(weights_file);
}
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/lc0ctl/describenet.cc

// begin of /Users/syys/CLionProjects/lc0/src/lc0ctl/leela2onnx.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const OptionId kInputFilenameId{"input", "InputFile",
                                "Path of the input Lc0 weights file."};
const OptionId kOutputFilenameId{"output", "OutputFile",
                                 "Path of the output ONNX file."};
const OptionId kInputPlanesName{"input-planes-name", "InputPlanesName",
                                "ONNX name to use for the input planes node."};
const OptionId kOutputPolicyHead{
    "policy-head-name", "PolicyHeadName",
    "ONNX name to use for the policy head output node."};
const OptionId kOutputWdl{"wdl-head-name", "WdlHeadName",
                          "ONNX name to use for the WDL head output node."};
const OptionId kOutputValue{
    "value-head-name", "ValueHeadName",
    "ONNX name to use for value policy head output node."};
const OptionId kOutputMlh{"mlh-head-name", "MlhHeadName",
                          "ONNX name to use for the MLH head output node."};
bool ProcessParameters(OptionsParser* options) {
  options->Add<StringOption>(kInputFilenameId);
  options->Add<StringOption>(kOutputFilenameId);
  options->Add<StringOption>(kInputPlanesName) = "/input/planes";
  options->Add<StringOption>(kOutputPolicyHead) = "/output/policy";
  options->Add<StringOption>(kOutputWdl) = "/output/wdl";
  options->Add<StringOption>(kOutputValue) = "/output/value";
  options->Add<StringOption>(kOutputMlh) = "/output/mlh";
  if (!options->ProcessAllFlags()) return false;
  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  dict.EnsureExists<std::string>(kOutputFilenameId);
  return true;
}
}  // namespace
void ConvertLeelaToOnnx() {
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;
  const OptionsDict& dict = options_parser.GetOptionsDict();
  auto weights_file =
      LoadWeightsFromFile(dict.Get<std::string>(kInputFilenameId));
  ShowNetworkFormatInfo(weights_file);
  if (weights_file.has_onnx_model()) {
    COUT << "The leela network already has ONNX network embedded, extracting.";
  } else {
    ShowNetworkWeightsInfo(weights_file);
    COUT << "Converting Leela network to the ONNX.";
    WeightsToOnnxConverterOptions onnx_options;
    onnx_options.input_planes_name = dict.Get<std::string>(kInputPlanesName);
    onnx_options.output_policy_head = dict.Get<std::string>(kOutputPolicyHead);
    onnx_options.output_wdl = dict.Get<std::string>(kOutputWdl);
    onnx_options.output_value = dict.Get<std::string>(kOutputValue);
    onnx_options.output_wdl = dict.Get<std::string>(kOutputWdl);
    weights_file = ConvertWeightsToOnnx(weights_file, onnx_options);
  }
  const auto& onnx = weights_file.onnx_model();
  WriteStringToFile(dict.Get<std::string>(kOutputFilenameId), onnx.model());
  ShowNetworkOnnxInfo(weights_file, false);
  COUT << "Done.";
}
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/lc0ctl/leela2onnx.cc

// begin of /Users/syys/CLionProjects/lc0/src/lc0ctl/onnx2leela.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
template <typename T, size_t N>
std::vector<std::string> GetAllEnumValues(const std::array<T, N>& vals,
                                          std::string (*func)(T)) {
  std::vector<std::string> res;
  std::transform(vals.begin(), vals.end(), std::back_inserter(res), func);
  return res;
}
template <typename T, size_t N>
T GetEnumValueFromString(const std::string& str_value,
                         const std::array<T, N>& vals, std::string (*func)(T)) {
  auto iter = std::find_if(vals.begin(), vals.end(),
                           [&](T val) { return func(val) == str_value; });
  if (iter == vals.end()) {
    throw Exception("Enum value " + str_value + " is unknown.");
  }
  return *iter;
}
const OptionId kInputFilenameId{"input", "InputFile",
                                "Path of the input Lc0 weights file."};
const OptionId kOutputFilenameId{"output", "OutputFile",
                                 "Path of the output ONNX file."};
const OptionId kInputFormatId(
    "input-format", "InputFormat",
    "Format in which the neural network expects input to be.");
const OptionId kPolicyFormatId(
    "policy-format", "PolicyFormat",
    "Format of the policy head output. Currently the search code does not "
    "distinguish between POLICY_CLASSICAL and POLICY_CONVOLUTION, but maybe "
    "one day for new types it will have new values.");
const OptionId kValueFormatId(
    "value-format", "ValueFormat",
    "Format of the value head output. Currently the search code does not "
    "distinguish between VALUE_CLASSICAL and VALUE_WDL, but maybe one day for "
    "new types it will have new values.");
const OptionId kMovesLeftFormatId("moves-left-format", "MovesLeftFormat",
                                  "Format of the moves left head output.");
// ONNX options.
const OptionId kOnnxDataTypeId("onnx-data-type", "OnnxDataType",
                               "Data type to feed into the neural network.");
const OptionId kOnnxInputId{"onnx-input", "OnnxInput",
                            "The name of the input ONNX node."};
const OptionId kOnnxOutputValueId{
    "onnx-output-value", "OnnxOutputValue",
    "The name of the node for a classical value head."};
const OptionId kOnnxOutputWdlId{"onnx-output-wdl", "OnnxOutputWdl",
                                "The name of the node for a wdl value head."};
const OptionId kOnnxOutputPolicyId{"onnx-output-policy", "OnnxOutputPolicy",
                                   "The name of the node for a policy head."};
const OptionId kOnnxOutputMlhId{"onnx-output-mlh", "OnnxOutputMlh",
                                "The name of the node for a moves left head."};
const OptionId kValidateModelId{"validate-weights", "ValidateWeights",
                                "Do a basic check of the provided ONNX file."};
bool ProcessParameters(OptionsParser* options) {
  using pblczero::NetworkFormat;
  using pblczero::OnnxModel;
  options->Add<StringOption>(kInputFilenameId);
  options->Add<StringOption>(kOutputFilenameId);
  // Data format options.
  options->Add<ChoiceOption>(
      kInputFormatId, GetAllEnumValues(NetworkFormat::InputFormat_AllValues,
                                       NetworkFormat::InputFormat_Name)) =
      NetworkFormat::InputFormat_Name(NetworkFormat::INPUT_CLASSICAL_112_PLANE);
  options->Add<ChoiceOption>(
      kPolicyFormatId, GetAllEnumValues(NetworkFormat::PolicyFormat_AllValues,
                                        NetworkFormat::PolicyFormat_Name)) =
      NetworkFormat::PolicyFormat_Name(NetworkFormat::POLICY_CLASSICAL);
  options->Add<ChoiceOption>(
      kValueFormatId, GetAllEnumValues(NetworkFormat::ValueFormat_AllValues,
                                       NetworkFormat::ValueFormat_Name)) =
      NetworkFormat::ValueFormat_Name(NetworkFormat::VALUE_WDL);
  options->Add<ChoiceOption>(
      kMovesLeftFormatId,
      GetAllEnumValues(NetworkFormat::MovesLeftFormat_AllValues,
                       NetworkFormat::MovesLeftFormat_Name)) =
      NetworkFormat::MovesLeftFormat_Name(NetworkFormat::MOVES_LEFT_V1);
  // Onnx options.
  options->Add<ChoiceOption>(kOnnxDataTypeId,
                             GetAllEnumValues(OnnxModel::DataType_AllValues,
                                              OnnxModel::DataType_Name)) =
      OnnxModel::DataType_Name(OnnxModel::FLOAT);
  options->Add<StringOption>(kOnnxInputId);
  options->Add<StringOption>(kOnnxOutputPolicyId);
  options->Add<StringOption>(kOnnxOutputValueId);
  options->Add<StringOption>(kOnnxOutputWdlId);
  options->Add<StringOption>(kOnnxOutputMlhId);
  options->Add<BoolOption>(kValidateModelId) = true;
  if (!options->ProcessAllFlags()) return false;
  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  dict.EnsureExists<std::string>(kOutputFilenameId);
  return true;
}
bool ValidateNetwork(const pblczero::Net& weights) {
  const auto& onnx_model = weights.onnx_model();
  pblczero::ModelProto onnx;
  onnx.ParseFromString(onnx_model.model());
  if (!onnx.has_ir_version()) {
    CERR << "ONNX file doesn't appear to have version specified. Likely not an "
            "ONNX file.";
    return false;
  }
  if (!onnx.has_domain()) {
    CERR << "ONNX file doesn't appear to have domain specified. Likely not an "
            "ONNX file.";
    return false;
  }
  const auto& onnx_inputs = onnx.graph().input();
  std::set<std::string> inputs;
  std::transform(onnx_inputs.begin(), onnx_inputs.end(),
                 std::inserter(inputs, inputs.end()),
                 [](const auto& x) { return std::string(x.name()); });
  const auto& onnx_outputs = onnx.graph().output();
  std::set<std::string> outputs;
  std::transform(onnx_outputs.begin(), onnx_outputs.end(),
                 std::inserter(outputs, outputs.end()),
                 [](const auto& x) { return std::string(x.name()); });
  auto check_exists = [](std::string_view n, std::set<std::string>* nodes) {
    std::string name(n);
    if (nodes->count(name) == 0) {
      CERR << "Node '" << name << "' doesn't exist in ONNX.";
      return false;
    }
    nodes->erase(name);
    return true;
  };
  if (onnx_model.has_input_planes() &&
      !check_exists(onnx_model.input_planes(), &inputs)) {
    return false;
  }
  if (onnx_model.has_output_value() &&
      !check_exists(onnx_model.output_value(), &outputs)) {
    return false;
  }
  if (onnx_model.has_output_wdl() &&
      !check_exists(onnx_model.output_wdl(), &outputs)) {
    return false;
  }
  if (onnx_model.has_output_policy() &&
      !check_exists(onnx_model.output_policy(), &outputs)) {
    return false;
  }
  if (onnx_model.has_output_mlh() &&
      !check_exists(onnx_model.output_mlh(), &outputs)) {
    return false;
  }
  for (const auto& input : inputs) {
    CERR << "Warning: ONNX input node '" << input << "' not used.";
  }
  for (const auto& output : outputs) {
    CERR << "Warning: ONNX output node '" << output << "' not used.";
  }
  if (!onnx_model.has_input_planes()) {
    CERR << "The --" << kOnnxInputId.long_flag()
         << " must be defined. Typical value for the ONNX networks exported "
            "from Leela is /input/planes.";
    return false;
  }
  if (!onnx_model.has_output_policy()) {
    CERR << "The --" << kOnnxOutputPolicyId.long_flag()
         << " must be defined. Typical value for the ONNX networks exported "
            "from Leela is /output/policy.";
    return false;
  }
  if (!onnx_model.has_output_value() && !onnx_model.has_output_wdl()) {
    CERR << "Either --" << kOnnxOutputValueId.long_flag() << " or --"
         << kOnnxOutputWdlId.long_flag()
         << " must be defined. Typical values for the ONNX networks exported "
            "from Leela are /output/value and /output/wdl.";
    return false;
  }
  if (onnx_model.has_output_value() && onnx_model.has_output_wdl()) {
    CERR << "Both --" << kOnnxOutputValueId.long_flag() << " and --"
         << kOnnxOutputWdlId.long_flag()
         << " are set. Only one of them has to be set.";
    return false;
  }
  return true;
}
}  // namespace
void ConvertOnnxToLeela() {
  using pblczero::NetworkFormat;
  using pblczero::OnnxModel;
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;
  const OptionsDict& dict = options_parser.GetOptionsDict();
  pblczero::Net out_weights;
  out_weights.set_magic(0x1c0);
  // ONNX networks appeared in v0.28.
  out_weights.mutable_min_version()->set_major(0);
  out_weights.mutable_min_version()->set_minor(28);
  auto format = out_weights.mutable_format()->mutable_network_format();
  format->set_network(NetworkFormat::NETWORK_ONNX);
  auto onnx = out_weights.mutable_onnx_model();
  onnx->set_data_type(GetEnumValueFromString(
      dict.Get<std::string>(kOnnxDataTypeId), OnnxModel::DataType_AllValues,
      OnnxModel::DataType_Name));
  // Input.
  format->set_input(GetEnumValueFromString(
      dict.Get<std::string>(kInputFormatId),
      NetworkFormat::InputFormat_AllValues, NetworkFormat::InputFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxInputId)) {
    onnx->set_input_planes(dict.Get<std::string>(kOnnxInputId));
  }
  // Policy.
  format->set_policy(GetEnumValueFromString(
      dict.Get<std::string>(kPolicyFormatId),
      NetworkFormat::PolicyFormat_AllValues, NetworkFormat::PolicyFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxOutputPolicyId)) {
    onnx->set_output_policy(dict.Get<std::string>(kOnnxOutputPolicyId));
  }
  // Value.
  format->set_value(GetEnumValueFromString(
      dict.Get<std::string>(kValueFormatId),
      NetworkFormat::ValueFormat_AllValues, NetworkFormat::ValueFormat_Name));
  if (dict.OwnExists<std::string>(kOnnxOutputValueId)) {
    onnx->set_output_value(dict.Get<std::string>(kOnnxOutputValueId));
  }
  if (dict.OwnExists<std::string>(kOnnxOutputWdlId)) {
    onnx->set_output_wdl(dict.Get<std::string>(kOnnxOutputWdlId));
  }
  // Mlh.
  if (dict.OwnExists<std::string>(kOnnxOutputMlhId)) {
    format->set_moves_left(
        GetEnumValueFromString(dict.Get<std::string>(kMovesLeftFormatId),
                               NetworkFormat::MovesLeftFormat_AllValues,
                               NetworkFormat::MovesLeftFormat_Name));
    onnx->set_output_mlh(dict.Get<std::string>(kOnnxOutputMlhId));
  }
  onnx->set_model(ReadFileToString(dict.Get<std::string>(kInputFilenameId)));
  if (dict.Get<bool>(kValidateModelId) && !ValidateNetwork(out_weights)) {
    return;
  }
  WriteStringToGzFile(dict.Get<std::string>(kOutputFilenameId),
                      out_weights.OutputAsString());
  ShowNetworkFormatInfo(out_weights);
  ShowNetworkOnnxInfo(out_weights, dict.Get<bool>(kValidateModelId));
  COUT << "Done.";
}
}  // namespace lczero
// end of /Users/syys/CLionProjects/lc0/src/lc0ctl/onnx2leela.cc

// begin of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/stoppers.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
///////////////////////////
// ChainedSearchStopper
///////////////////////////
bool ChainedSearchStopper::ShouldStop(const IterationStats& stats,
                                      StoppersHints* hints) {
  for (const auto& x : stoppers_) {
    if (x->ShouldStop(stats, hints)) return true;
  }
  return false;
}
void ChainedSearchStopper::AddStopper(std::unique_ptr<SearchStopper> stopper) {
  if (stopper) stoppers_.push_back(std::move(stopper));
}
void ChainedSearchStopper::OnSearchDone(const IterationStats& stats) {
  for (const auto& x : stoppers_) x->OnSearchDone(stats);
}
///////////////////////////
// VisitsStopper
///////////////////////////
bool VisitsStopper::ShouldStop(const IterationStats& stats,
                               StoppersHints* hints) {
  if (populate_remaining_playouts_) {
    hints->UpdateEstimatedRemainingPlayouts(nodes_limit_ - stats.total_nodes);
  }
  if (stats.total_nodes >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached visits limit: " << stats.total_nodes
            << ">=" << nodes_limit_;
    return true;
  }
  return false;
}
///////////////////////////
// PlayoutsStopper
///////////////////////////
bool PlayoutsStopper::ShouldStop(const IterationStats& stats,
                                 StoppersHints* hints) {
  if (populate_remaining_playouts_) {
    hints->UpdateEstimatedRemainingPlayouts(nodes_limit_ -
                                            stats.nodes_since_movestart);
  }
  if (stats.nodes_since_movestart >= nodes_limit_) {
    LOGFILE << "Stopped search: Reached playouts limit: "
            << stats.nodes_since_movestart << ">=" << nodes_limit_;
    return true;
  }
  return false;
}
///////////////////////////
// MemoryWatchingStopper
///////////////////////////
namespace {
const size_t kAvgNodeSize =
    sizeof(Node) + MemoryWatchingStopper::kAvgMovesPerPosition * sizeof(Edge);
const size_t kAvgCacheItemSize =
    NNCache::GetItemStructSize() + sizeof(CachedNNRequest) +
    sizeof(CachedNNRequest::IdxAndProb) *
        MemoryWatchingStopper::kAvgMovesPerPosition;
}  // namespace
MemoryWatchingStopper::MemoryWatchingStopper(int cache_size, int ram_limit_mb,
                                             bool populate_remaining_playouts)
    : VisitsStopper(
          (ram_limit_mb * 1000000LL - cache_size * kAvgCacheItemSize) /
              kAvgNodeSize,
          populate_remaining_playouts) {
  LOGFILE << "RAM limit " << ram_limit_mb << "MB. Cache takes "
          << cache_size * kAvgCacheItemSize / 1000000
          << "MB. Remaining memory is enough for " << GetVisitsLimit()
          << " nodes.";
}
///////////////////////////
// TimelimitStopper
///////////////////////////
TimeLimitStopper::TimeLimitStopper(int64_t time_limit_ms)
    : time_limit_ms_(time_limit_ms) {}
bool TimeLimitStopper::ShouldStop(const IterationStats& stats,
                                  StoppersHints* hints) {
  hints->UpdateEstimatedRemainingTimeMs(time_limit_ms_ -
                                        stats.time_since_movestart);
  if (stats.time_since_movestart >= time_limit_ms_) {
    LOGFILE << "Stopping search: Ran out of time.";
    return true;
  }
  return false;
}
int64_t TimeLimitStopper::GetTimeLimitMs() const { return time_limit_ms_; }
///////////////////////////
// DepthStopper
///////////////////////////
bool DepthStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  if (stats.average_depth >= depth_) {
    LOGFILE << "Stopped search: Reached depth.";
    return true;
  }
  return false;
}
///////////////////////////
// KldGainStopper
///////////////////////////
KldGainStopper::KldGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}
bool KldGainStopper::ShouldStop(const IterationStats& stats, StoppersHints*) {
  Mutex::Lock lock(mutex_);
  const auto new_child_nodes = stats.total_nodes - 1.0;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;
  const auto new_visits = stats.edge_n;
  if (!prev_visits_.empty()) {
    double kldgain = 0.0;
    for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = new_visits[i] / new_child_nodes;
      if (prev_visits_[i] != 0) kldgain += o_p * log(o_p / n_p);
    }
    if (kldgain / (new_child_nodes - prev_child_nodes_) < min_gain_) {
      LOGFILE << "Stopping search: KLDGain per node too small.";
      return true;
    }
  }
  prev_visits_ = new_visits;
  prev_child_nodes_ = new_child_nodes;
  return false;
}
///////////////////////////
// SmartPruningStopper
///////////////////////////
namespace {
const int kSmartPruningToleranceMs = 200;
const int kSmartPruningToleranceNodes = 300;
}  // namespace
SmartPruningStopper::SmartPruningStopper(float smart_pruning_factor,
                                         int64_t minimum_batches)
    : smart_pruning_factor_(smart_pruning_factor),
      minimum_batches_(minimum_batches) {}
bool SmartPruningStopper::ShouldStop(const IterationStats& stats,
                                     StoppersHints* hints) {
  if (smart_pruning_factor_ <= 0.0) return false;
  Mutex::Lock lock(mutex_);
  if (stats.edge_n.size() == 1) {
    LOGFILE << "Only one possible move. Moving immediately.";
    return true;
  }
  if (stats.edge_n.size() <= static_cast<size_t>(stats.num_losing_edges + 1)) {
    LOGFILE << "At most one non losing move, stopping search.";
    return true;
  }
  if (stats.win_found) {
    LOGFILE << "Terminal win found, stopping search.";
    return true;
  }
  if (stats.nodes_since_movestart > 0 && !first_eval_time_) {
    first_eval_time_ = stats.time_since_movestart;
    return false;
  }
  if (!first_eval_time_) return false;
  if (stats.edge_n.size() == 0) return false;
  if (stats.time_since_movestart <
      *first_eval_time_ + kSmartPruningToleranceMs) {
    return false;
  }
  const auto nodes = stats.nodes_since_movestart + kSmartPruningToleranceNodes;
  const auto time = stats.time_since_movestart - *first_eval_time_;
  // If nps is populated by someone who knows better, use it. Otherwise use the
  // value calculated here.
  const auto nps = hints->GetEstimatedNps().value_or(1000LL * nodes / time + 1);
  const double remaining_time_s = hints->GetEstimatedRemainingTimeMs() / 1000.0;
  const auto remaining_playouts =
      std::min(remaining_time_s * nps / smart_pruning_factor_,
               hints->GetEstimatedRemainingPlayouts() / smart_pruning_factor_);
  // May overflow if (nps/smart_pruning_factor) > 180 000 000, but that's not
  // very realistic.
  hints->UpdateEstimatedRemainingPlayouts(remaining_playouts);
  if (stats.batches_since_movestart < minimum_batches_) return false;
  uint32_t largest_n = 0;
  uint32_t second_largest_n = 0;
  for (auto n : stats.edge_n) {
    if (n > largest_n) {
      second_largest_n = largest_n;
      largest_n = n;
    } else if (n > second_largest_n) {
      second_largest_n = n;
    }
  }
  if (remaining_playouts < (largest_n - second_largest_n)) {
    LOGFILE << remaining_playouts << " playouts remaining. Best move has "
            << largest_n << " visits, second best -- " << second_largest_n
            << ". Difference is " << (largest_n - second_largest_n)
            << ", so stopping the search after "
            << stats.batches_since_movestart << " batches.";
    return true;
  }
  return false;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/mcts/stoppers/stoppers.cc

// begin of /Users/syys/CLionProjects/lc0/src/trainingdata/writer.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
std::string GetLc0CacheDirectory() {
  std::string user_cache_path = GetUserCacheDirectory();
  if (!user_cache_path.empty()) {
    user_cache_path += "lc0/";
    CreateDirectory(user_cache_path);
  }
  return user_cache_path;
}
}  // namespace
TrainingDataWriter::TrainingDataWriter(int game_id) {
  static std::string directory =
      GetLc0CacheDirectory() + "data-" + Random::Get().GetString(12);
  // It's fine if it already exists.
  CreateDirectory(directory.c_str());
  std::ostringstream oss;
  oss << directory << '/' << "game_" << std::setfill('0') << std::setw(6)
      << game_id << ".gz";
  filename_ = oss.str();
  fout_ = gzopen(filename_.c_str(), "wb");
  if (!fout_) throw Exception("Cannot create gzip file " + filename_);
}
void TrainingDataWriter::WriteChunk(const V6TrainingData& data) {
  auto bytes_written =
      gzwrite(fout_, reinterpret_cast<const char*>(&data), sizeof(data));
  if (bytes_written != sizeof(data)) {
    throw Exception("Unable to write into " + filename_);
  }
}
void TrainingDataWriter::Finalize() {
  gzclose(fout_);
  fout_ = nullptr;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/trainingdata/writer.cc

// begin of /Users/syys/CLionProjects/lc0/src/trainingdata/trainingdata.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
void DriftCorrect(float* q, float* d) {
  // Training data doesn't have a high number of nodes, so there shouldn't be
  // too much drift. Highest known value not caused by backend bug was 1.5e-7.
  const float allowed_eps = 0.000001f;
  if (*q > 1.0f) {
    if (*q > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in q " << *q;
    }
    *q = 1.0f;
  }
  if (*q < -1.0f) {
    if (*q < -1.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in q " << *q;
    }
    *q = -1.0f;
  }
  if (*d > 1.0f) {
    if (*d > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in d " << *d;
    }
    *d = 1.0f;
  }
  if (*d < 0.0f) {
    if (*d < 0.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in d " << *d;
    }
    *d = 0.0f;
  }
  float w = (1.0f - *d + *q) / 2.0f;
  float l = w - *q;
  // Assume q drift is rarer than d drift and apply all correction to d.
  if (w < 0.0f || l < 0.0f) {
    float drift = 2.0f * std::min(w, l);
    if (drift < -allowed_eps) {
      CERR << "Unexpectedly large drift correction for d based on q. " << drift;
    }
    *d += drift;
    // Since q is in range -1 to 1 - this correction should never push d outside
    // of range, but precision could be lost in calculations so just in case.
    if (*d < 0.0f) {
      *d = 0.0f;
    }
  }
}
}  // namespace
void V6TrainingDataArray::Write(TrainingDataWriter* writer, GameResult result,
                                bool adjudicated) const {
  if (training_data_.empty()) return;
  // Base estimate off of best_m.  If needed external processing can use a
  // different approach.
  float m_estimate = training_data_.back().best_m + training_data_.size() - 1;
  for (auto chunk : training_data_) {
    bool black_to_move = chunk.side_to_move_or_enpassant;
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            chunk.input_format))) {
      black_to_move = (chunk.invariance_info & (1u << 7)) != 0;
    }
    if (result == GameResult::WHITE_WON) {
      chunk.result_q = black_to_move ? -1 : 1;
      chunk.result_d = 0;
    } else if (result == GameResult::BLACK_WON) {
      chunk.result_q = black_to_move ? 1 : -1;
      chunk.result_d = 0;
    } else {
      chunk.result_q = 0;
      chunk.result_d = 1;
    }
    if (adjudicated) {
      chunk.invariance_info |= 1u << 5;  // Game adjudicated.
    }
    if (adjudicated && result == GameResult::UNDECIDED) {
      chunk.invariance_info |= 1u << 4;  // Max game length exceeded.
    }
    chunk.plies_left = m_estimate;
    m_estimate -= 1.0f;
    writer->WriteChunk(chunk);
  }
}
void V6TrainingDataArray::Add(const Node* node, const PositionHistory& history,
                              Eval best_eval, Eval played_eval,
                              bool best_is_proven, Move best_move,
                              Move played_move, const NNCacheLock& nneval) {
  V6TrainingData result;
  const auto& position = history.Last();
  // Set version.
  result.version = 6;
  result.input_format = input_format_;
  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(
      input_format_, history, 8, fill_empty_history_[position.IsBlackToMove()],
      &transform);
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }
  // Populate probabilities.
  auto total_n = node->GetChildrenVisits();
  // Prevent garbage/invalid training data from being uploaded to server.
  // It's possible to have N=0 when there is only one legal move in position
  // (due to smart pruning).
  if (total_n == 0 && node->GetNumEdges() != 1) {
    throw Exception("Search generated invalid data!");
  }
  // Set illegal moves to have -1 probability.
  std::fill(std::begin(result.probabilities), std::end(result.probabilities),
            -1);
  // Set moves probabilities according to their relative amount of visits.
  // Compute Kullback-Leibler divergence in nats (between policy and visits).
  float kld_sum = 0;
  float max_p = -std::numeric_limits<float>::infinity();
  std::vector<float> intermediate;
  if (nneval) {
    int last_idx = 0;
    for (const auto& child : node->Edges()) {
      auto nn_idx = child.edge()->GetMove().as_nn_index(transform);
      float p = 0;
      for (int i = 0; i < nneval->p.size(); i++) {
        // Optimization: usually moves are stored in the same order as queried.
        const auto& move = nneval->p[last_idx++];
        if (last_idx == nneval->p.size()) last_idx = 0;
        if (move.first == nn_idx) {
          p = move.second;
          break;
        }
      }
      intermediate.emplace_back(p);
      max_p = std::max(max_p, p);
    }
  }
  float total = 0.0;
  auto it = intermediate.begin();
  for (const auto& child : node->Edges()) {
    auto nn_idx = child.edge()->GetMove().as_nn_index(transform);
    float fracv = total_n > 0 ? child.GetN() / static_cast<float>(total_n) : 1;
    if (nneval) {
      float P = std::exp(*it - max_p);
      if (fracv > 0) {
        kld_sum += fracv * std::log(fracv / P);
      }
      total += P;
      it++;
    }
    result.probabilities[nn_idx] = fracv;
  }
  if (nneval) {
    // Add small epsilon for backward compatibility with earlier value of 0.
    auto epsilon = std::numeric_limits<float>::min();
    kld_sum = std::max(kld_sum + std::log(total), 0.0f) + epsilon;
  }
  result.policy_kld = kld_sum;
  const auto& castlings = position.GetBoard().castlings();
  // Populate castlings.
  // For non-frc trained nets, just send 1 like we used to.
  uint8_t queen_side = 1;
  uint8_t king_side = 1;
  // If frc trained, send the bit mask representing rook position.
  if (Is960CastlingFormat(input_format_)) {
    queen_side <<= castlings.queenside_rook();
    king_side <<= castlings.kingside_rook();
  }
  result.castling_us_ooo = castlings.we_can_000() ? queen_side : 0;
  result.castling_us_oo = castlings.we_can_00() ? king_side : 0;
  result.castling_them_ooo = castlings.they_can_000() ? queen_side : 0;
  result.castling_them_oo = castlings.they_can_00() ? king_side : 0;
  // Other params.
  if (IsCanonicalFormat(input_format_)) {
    result.side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      result.side_to_move_or_enpassant =
          ReverseBitsInBytes(result.side_to_move_or_enpassant);
    }
    // Send transform in deprecated move count so rescorer can reverse it to
    // calculate the actual move list from the input data.
    result.invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    result.side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    result.invariance_info = 0;
  }
  if (best_is_proven) {
    result.invariance_info |= 1u << 3;  // Best node is proven best;
  }
  result.dummy = 0;
  result.rule50_count = position.GetRule50Ply();
  // Game result is undecided.
  result.result_q = 0;
  result.result_d = 1;
  Eval orig_eval;
  if (nneval) {
    orig_eval.wl = nneval->q;
    orig_eval.d = nneval->d;
    orig_eval.ml = nneval->m;
  } else {
    orig_eval.wl = std::numeric_limits<float>::quiet_NaN();
    orig_eval.d = std::numeric_limits<float>::quiet_NaN();
    orig_eval.ml = std::numeric_limits<float>::quiet_NaN();
  }
  // Aggregate evaluation WL.
  result.root_q = -node->GetWL();
  result.best_q = best_eval.wl;
  result.played_q = played_eval.wl;
  result.orig_q = orig_eval.wl;
  // Draw probability of WDL head.
  result.root_d = node->GetD();
  result.best_d = best_eval.d;
  result.played_d = played_eval.d;
  result.orig_d = orig_eval.d;
  DriftCorrect(&result.best_q, &result.best_d);
  DriftCorrect(&result.root_q, &result.root_d);
  DriftCorrect(&result.played_q, &result.played_d);
  result.root_m = node->GetM();
  result.best_m = best_eval.ml;
  result.played_m = played_eval.ml;
  result.orig_m = orig_eval.ml;
  result.visits = node->GetN();
  if (position.IsBlackToMove()) {
    best_move.Mirror();
    played_move.Mirror();
  }
  result.best_idx = best_move.as_nn_index(transform);
  result.played_idx = played_move.as_nn_index(transform);
  result.reserved = 0;
  // Unknown here - will be filled in once the full data has been collected.
  result.plies_left = 0;
  training_data_.push_back(result);
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/trainingdata/trainingdata.cc

// begin of /Users/syys/CLionProjects/lc0/src/selfplay/game.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const OptionId kReuseTreeId{"reuse-tree", "ReuseTree",
                            "Reuse the search tree between moves."};
const OptionId kResignPercentageId{
    "resign-percentage", "ResignPercentage",
    "Resign when win percentage drops below specified value."};
const OptionId kResignWDLStyleId{
    "resign-wdlstyle", "ResignWDLStyle",
    "If set, resign percentage applies to any output state being above "
    "100% minus the percentage instead of winrate being below."};
const OptionId kResignEarliestMoveId{"resign-earliest-move",
                                     "ResignEarliestMove",
                                     "Earliest move that resign is allowed."};
const OptionId kMinimumAllowedVistsId{
    "minimum-allowed-visits", "MinimumAllowedVisits",
    "Unless the selected move is the best move, temperature based selection "
    "will be retried until visits of selected move is greater than or equal to "
    "this threshold."};
const OptionId kUciChess960{
    "chess960", "UCI_Chess960",
    "Castling moves are encoded as \"king takes rook\"."};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
}  // namespace
void SelfPlayGame::PopulateUciParams(OptionsParser* options) {
  options->Add<BoolOption>(kReuseTreeId) = false;
  options->Add<BoolOption>(kResignWDLStyleId) = false;
  options->Add<FloatOption>(kResignPercentageId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kResignEarliestMoveId, 0, 1000) = 0;
  options->Add<IntOption>(kMinimumAllowedVistsId, 0, 1000000) = 0;
  options->Add<BoolOption>(kUciChess960) = false;
  PopulateTimeManagementOptions(RunType::kSelfplay, options);
  options->Add<StringOption>(kSyzygyTablebaseId);
}
SelfPlayGame::SelfPlayGame(PlayerOptions white, PlayerOptions black,
                           bool shared_tree, const Opening& opening)
    : options_{white, black},
      chess960_{white.uci_options->Get<bool>(kUciChess960) ||
                black.uci_options->Get<bool>(kUciChess960)},
      training_data_(SearchParams(*white.uci_options).GetHistoryFill(),
                     SearchParams(*black.uci_options).GetHistoryFill(),
                     white.network->GetCapabilities().input_format) {
  orig_fen_ = opening.start_fen;
  tree_[0] = std::make_shared<NodeTree>();
  tree_[0]->ResetToPosition(orig_fen_, {});
  if (shared_tree) {
    tree_[1] = tree_[0];
  } else {
    tree_[1] = std::make_shared<NodeTree>();
    tree_[1]->ResetToPosition(orig_fen_, {});
  }
  for (Move m : opening.moves) {
    tree_[0]->MakeMove(m);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(m);
  }
}
void SelfPlayGame::Play(int white_threads, int black_threads, bool training,
                        SyzygyTablebase* syzygy_tb, bool enable_resign) {
  bool blacks_move = tree_[0]->IsBlackToMove();
  // If we are training, verify that input formats are consistent.
  if (training && options_[0].network->GetCapabilities().input_format !=
      options_[1].network->GetCapabilities().input_format) {
    throw Exception("Can't mix networks with different input format!");
  }
  // Take syzygy tablebases from player1 options.
  std::string tb_paths =
      options_[0].uci_options->Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty()) {  // && tb_paths != tb_paths_) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
  }
  // Do moves while not end of the game. (And while not abort_)
  while (!abort_) {
    game_result_ = tree_[0]->GetPositionHistory().ComputeGameResult();
    // If endgame, stop.
    if (game_result_ != GameResult::UNDECIDED) break;
    if (tree_[0]->GetPositionHistory().Last().GetGamePly() >= 450) {
      adjudicated_ = true;
      break;
    }
    // Initialize search.
    const int idx = blacks_move ? 1 : 0;
    if (!options_[idx].uci_options->Get<bool>(kReuseTreeId)) {
      tree_[idx]->TrimTreeAtHead();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
      auto stoppers = options_[idx].search_limits.MakeSearchStopper();
      PopulateIntrinsicStoppers(stoppers.get(), *options_[idx].uci_options);
      std::unique_ptr<UciResponder> responder =
          std::make_unique<CallbackUciResponder>(
              options_[idx].best_move_callback, options_[idx].info_callback);
      if (!chess960_) {
        // Remap FRC castling to legacy castling.
        responder = std::make_unique<Chess960Transformer>(
            std::move(responder), tree_[idx]->HeadPosition().GetBoard());
      }
      search_ = std::make_unique<Search>(
          *tree_[idx], options_[idx].network, std::move(responder),
          /* searchmoves */ MoveList(), std::chrono::steady_clock::now(),
          std::move(stoppers),
          /* infinite */ false, *options_[idx].uci_options, options_[idx].cache,
          syzygy_tb);
    }
    // Do search.
    search_->RunBlocking(blacks_move ? black_threads : white_threads);
    move_count_++;
    nodes_total_ += search_->GetTotalPlayouts();
    if (abort_) break;
    Move best_move;
    bool best_is_terminal;
    const auto best_eval = search_->GetBestEval(&best_move, &best_is_terminal);
    float eval = best_eval.wl;
    eval = (eval + 1) / 2;
    if (eval < min_eval_[idx]) min_eval_[idx] = eval;
    const int move_number = tree_[0]->GetPositionHistory().GetLength() / 2 + 1;
    auto best_w = (best_eval.wl + 1.0f - best_eval.d) / 2.0f;
    auto best_d = best_eval.d;
    auto best_l = best_w - best_eval.wl;
    max_eval_[0] = std::max(max_eval_[0], blacks_move ? best_l : best_w);
    max_eval_[1] = std::max(max_eval_[1], best_d);
    max_eval_[2] = std::max(max_eval_[2], blacks_move ? best_w : best_l);
    if (enable_resign &&
        move_number >=
            options_[idx].uci_options->Get<int>(kResignEarliestMoveId)) {
      const float resignpct =
          options_[idx].uci_options->Get<float>(kResignPercentageId) / 100;
      if (options_[idx].uci_options->Get<bool>(kResignWDLStyleId)) {
        auto threshold = 1.0f - resignpct;
        if (best_w > threshold) {
          game_result_ =
              blacks_move ? GameResult::BLACK_WON : GameResult::WHITE_WON;
          adjudicated_ = true;
          break;
        }
        if (best_l > threshold) {
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
        if (best_d > threshold) {
          game_result_ = GameResult::DRAW;
          adjudicated_ = true;
          break;
        }
      } else {
        if (eval < resignpct) {  // always false when resignpct == 0
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
      }
    }
    auto node = tree_[idx]->GetCurrentHead();
    Eval played_eval = best_eval;
    Move move;
    while (true) {
      move = search_->GetBestMove().first;
      uint32_t max_n = 0;
      uint32_t cur_n = 0;
      for (auto& edge : node->Edges()) {
        if (edge.GetN() > max_n) {
          max_n = edge.GetN();
        }
        if (edge.GetMove(tree_[idx]->IsBlackToMove()) == move) {
          cur_n = edge.GetN();
          played_eval.wl = edge.GetWL(-node->GetWL());
          played_eval.d = edge.GetD(node->GetD());
          played_eval.ml = edge.GetM(node->GetM() - 1) + 1;
        }
      }
      // If 'best move' is less than allowed visits and not max visits,
      // discard it and try again.
      if (cur_n == max_n ||
          static_cast<int>(cur_n) >=
              options_[idx].uci_options->Get<int>(kMinimumAllowedVistsId)) {
        break;
      }
      PositionHistory history_copy = tree_[idx]->GetPositionHistory();
      Move move_for_history = move;
      if (tree_[idx]->IsBlackToMove()) {
        move_for_history.Mirror();
      }
      history_copy.Append(move_for_history);
      // Ensure not to discard games that are already decided.
      if (history_copy.ComputeGameResult() == GameResult::UNDECIDED) {
        auto move_list_to_discard = GetMoves();
        move_list_to_discard.push_back(move);
        options_[idx].discarded_callback({orig_fen_, move_list_to_discard});
      }
      search_->ResetBestMove();
    }
    if (training) {
      bool best_is_proof = best_is_terminal;  // But check for better moves.
      if (best_is_proof && best_eval.wl < 1) {
        auto best =
            (best_eval.wl == 0) ? GameResult::DRAW : GameResult::BLACK_WON;
        auto upper = best;
        for (const auto& edge : node->Edges()) {
          upper = std::max(edge.GetBounds().second, upper);
        }
        if (best < upper) {
          best_is_proof = false;
        }
      }
      // Append training data. The GameResult is later overwritten.
      NNCacheLock nneval =
          search_->GetCachedNNEval(tree_[idx]->GetCurrentHead());
      training_data_.Add(tree_[idx]->GetCurrentHead(),
                         tree_[idx]->GetPositionHistory(), best_eval,
                         played_eval, best_is_proof, best_move, move, nneval);
    }
    // Must reset the search before mutating the tree.
    search_.reset();
    // Add best move to the tree.
    tree_[0]->MakeMove(move);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(move);
    blacks_move = !blacks_move;
  }
}
std::vector<Move> SelfPlayGame::GetMoves() const {
  std::vector<Move> moves;
  for (Node* node = tree_[0]->GetCurrentHead();
       node != tree_[0]->GetGameBeginNode(); node = node->GetParent()) {
    moves.push_back(node->GetParent()->GetEdgeToNode(node)->GetMove());
  }
  std::vector<Move> result;
  Position pos = tree_[0]->GetPositionHistory().Starting();
  while (!moves.empty()) {
    Move move = moves.back();
    moves.pop_back();
    if (!chess960_) move = pos.GetBoard().GetLegacyMove(move);
    pos = Position(pos, move);
    // Position already flipped, therefore flip the move if white to move.
    if (!pos.IsBlackToMove()) move.Mirror();
    result.push_back(move);
  }
  return result;
}
float SelfPlayGame::GetWorstEvalForWinnerOrDraw() const {
  // TODO: This assumes both players have the same resign style.
  // Supporting otherwise involves mixing the meaning of worst.
  if (options_[0].uci_options->Get<bool>(kResignWDLStyleId)) {
    if (game_result_ == GameResult::WHITE_WON) {
      return std::max(max_eval_[1], max_eval_[2]);
    } else if (game_result_ == GameResult::BLACK_WON) {
      return std::max(max_eval_[1], max_eval_[0]);
    } else {
      return std::max(max_eval_[2], max_eval_[0]);
    }
  }
  if (game_result_ == GameResult::WHITE_WON) return min_eval_[0];
  if (game_result_ == GameResult::BLACK_WON) return min_eval_[1];
  return std::min(min_eval_[0], min_eval_[1]);
}
void SelfPlayGame::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
  if (search_) search_->Abort();
}
void SelfPlayGame::WriteTrainingData(TrainingDataWriter* writer) const {
  training_data_.Write(writer, game_result_, adjudicated_);
}
std::unique_ptr<ChainedSearchStopper> SelfPlayLimits::MakeSearchStopper()
    const {
  auto result = std::make_unique<ChainedSearchStopper>();
  // always set VisitsStopper to avoid exceeding the limit 4000000000, the
  // default value when visits = 0
  result->AddStopper(std::make_unique<VisitsStopper>(visits, false));
  if (playouts >= 0) {
    result->AddStopper(std::make_unique<PlayoutsStopper>(playouts, false));
  }
  if (movetime >= 0) {
    result->AddStopper(std::make_unique<TimeLimitStopper>(movetime));
  }
  return result;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/selfplay/game.cc

// begin of /Users/syys/CLionProjects/lc0/src/selfplay/tournament.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const OptionId kShareTreesId{"share-trees", "ShareTrees",
                             "When on, game tree is shared for two players; "
                             "when off, each side has a separate tree."};
const OptionId kTotalGamesId{
    "games", "Games",
    "Number of games to play. -1 to play forever, -2 to play equal to book "
    "length, or double book length if mirrored."};
const OptionId kParallelGamesId{"parallelism", "Parallelism",
                                "Number of games to play in parallel."};
const OptionId kThreadsId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use for every game,", 't'};
const OptionId kPlayoutsId{"playouts", "Playouts",
                           "Number of playouts per move to search."};
const OptionId kVisitsId{"visits", "Visits",
                         "Number of visits per move to search."};
const OptionId kTimeMsId{"movetime", "MoveTime",
                         "Time per move, in milliseconds."};
const OptionId kTrainingId{
    "training", "Training",
    "Enables writing training data. The training data is stored into a "
    "temporary subdirectory that the engine creates."};
const OptionId kVerboseThinkingId{"verbose-thinking", "VerboseThinking",
                                  "Show verbose thinking messages."};
const OptionId kMoveThinkingId{"move-thinking", "MoveThinking",
                               "Show all the per-move thinking."};
const OptionId kResignPlaythroughId{
    "resign-playthrough", "ResignPlaythrough",
    "The percentage of games which ignore resign."};
const OptionId kDiscardedStartChanceId{
    "discarded-start-chance", "DiscardedStartChance",
    "The percentage chance each game will attempt to start from a position "
    "discarded due to not getting enough visits."};
const OptionId kOpeningsFileId{
    "openings-pgn", "OpeningsPgnFile",
    "A path name to a pgn file containing openings to use."};
const OptionId kOpeningsMirroredId{
    "mirror-openings", "MirrorOpenings",
    "If true, each opening will be played in pairs. "
    "Not really compatible with openings mode random."};
const OptionId kOpeningsModeId{"openings-mode", "OpeningsMode",
                               "A choice of sequential, shuffled, or random."};
const OptionId kSyzygyTablebaseId{
	"syzygy-paths", "SyzygyPath",
	"List of Syzygy tablebase directories, list entries separated by system "
	"separator (\";\" for Windows, \":\" for Linux).",
	's' };
}  // namespace
void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");
  options->AddContext("white");
  options->AddContext("black");
  for (const auto context : {"player1", "player2"}) {
    auto* dict = options->GetMutableOptions(context);
    dict->AddSubdict("white")->AddAliasDict(&options->GetOptionsDict("white"));
    dict->AddSubdict("black")->AddAliasDict(&options->GetOptionsDict("black"));
  }
  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  SearchParams::Populate(options);
  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -2, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<BoolOption>(kMoveThinkingId) = false;
  options->Add<FloatOption>(kResignPlaythroughId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kDiscardedStartChanceId, 0.0f, 100.0f) = 0.0f;
  options->Add<StringOption>(kOpeningsFileId) = "";
  options->Add<BoolOption>(kOpeningsMirroredId) = false;
  std::vector<std::string> openings_modes = {"sequential", "shuffled",
                                             "random"};
  options->Add<ChoiceOption>(kOpeningsModeId, openings_modes) = "sequential";
  options->Add<StringOption>(kSyzygyTablebaseId);
  SelfPlayGame::PopulateUciParams(options);
  auto defaults = options->GetMutableDefaultsOptions();
  defaults->Set<int>(SearchParams::kMiniBatchSizeId, 32);
  defaults->Set<float>(SearchParams::kCpuctId, 1.2f);
  defaults->Set<float>(SearchParams::kCpuctFactorId, 0.0f);
  defaults->Set<float>(SearchParams::kPolicySoftmaxTempId, 1.0f);
  defaults->Set<int>(SearchParams::kMaxCollisionVisitsId, 1);
  defaults->Set<int>(SearchParams::kMaxCollisionEventsId, 1);
  defaults->Set<int>(SearchParams::kCacheHistoryLengthId, 7);
  defaults->Set<bool>(SearchParams::kOutOfOrderEvalId, false);
  defaults->Set<float>(SearchParams::kTemperatureId, 1.0f);
  defaults->Set<float>(SearchParams::kNoiseEpsilonId, 0.25f);
  defaults->Set<float>(SearchParams::kFpuValueId, 0.0f);
  defaults->Set<std::string>(SearchParams::kHistoryFillId, "no");
  defaults->Set<std::string>(NetworkFactory::kBackendId, "multiplexing");
  defaults->Set<bool>(SearchParams::kStickyEndgamesId, false);
  defaults->Set<bool>(SearchParams::kTwoFoldDrawsId, false);
  defaults->Set<int>(SearchParams::kTaskWorkersPerSearchWorkerId, 0);
}
SelfPlayTournament::SelfPlayTournament(
    const OptionsDict& options,
    CallbackUciResponder::BestMoveCallback best_move_info,
    CallbackUciResponder::ThinkingCallback thinking_info,
    GameInfo::Callback game_info, TournamentInfo::Callback tournament_info)
    : player_options_{{options.GetSubdict("player1").GetSubdict("white"),
                       options.GetSubdict("player1").GetSubdict("black")},
                      {options.GetSubdict("player2").GetSubdict("white"),
                       options.GetSubdict("player2").GetSubdict("black")}},
      best_move_callback_(best_move_info),
      info_callback_(thinking_info),
      game_callback_(game_info),
      tournament_callback_(tournament_info),
      kTotalGames(options.Get<int>(kTotalGamesId)),
      kShareTree(options.Get<bool>(kShareTreesId)),
      kParallelism(options.Get<int>(kParallelGamesId)),
      kTraining(options.Get<bool>(kTrainingId)),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId)),
      kDiscardedStartChance(options.Get<float>(kDiscardedStartChanceId)) {
  std::string book = options.Get<std::string>(kOpeningsFileId);
  if (!book.empty()) {
    PgnReader book_reader;
    book_reader.AddPgnFile(book);
    openings_ = book_reader.ReleaseGames();
    if (options.Get<std::string>(kOpeningsModeId) == "shuffled") {
      Random::Get().Shuffle(openings_.begin(), openings_.end());
    }
  }
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    first_game_black_ = Random::Get().GetBool();
  }
  // Initializing networks.
  for (const auto& name : {"player1", "player2"}) {
    for (const auto& color : {"white", "black"}) {
      const auto& opts = options.GetSubdict(name).GetSubdict(color);
      const auto config = NetworkFactory::BackendConfiguration(opts);
      if (networks_.find(config) == networks_.end()) {
        networks_.emplace(config, NetworkFactory::LoadNetwork(opts));
      }
    }
  }
  // Initializing cache.
  cache_[0] = std::make_shared<NNCache>(
      options.GetSubdict("player1").Get<int>(kNNCacheSizeId));
  if (kShareTree) {
    cache_[1] = cache_[0];
  } else {
    cache_[1] = std::make_shared<NNCache>(
        options.GetSubdict("player2").Get<int>(kNNCacheSizeId));
  }
  // SearchLimits.
  static constexpr const char* kPlayerNames[2] = {"player1", "player2"};
  static constexpr const char* kPlayerColors[2] = {"white", "black"};
  for (int name_idx : {0, 1}) {
    for (int color_idx : {0, 1}) {
      auto& limits = search_limits_[name_idx][color_idx];
      const auto& dict = options.GetSubdict(kPlayerNames[name_idx])
                             .GetSubdict(kPlayerColors[color_idx]);
      limits.playouts = dict.Get<int>(kPlayoutsId);
      limits.visits = dict.Get<int>(kVisitsId);
      limits.movetime = dict.Get<int>(kTimeMsId);
      if (limits.playouts == -1 && limits.visits == -1 &&
          limits.movetime == -1) {
        throw Exception(
            "Please define --visits, --playouts or --movetime, otherwise it's "
            "not clear when to stop search.");
      }
    }
  }
  // Take syzygy tablebases from options.
  std::string tb_paths =
	  options.Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty()) {
	  syzygy_tb_ = std::make_unique<SyzygyTablebase>();
	  CERR << "Loading Syzygy tablebases from " << tb_paths;
	  if (!syzygy_tb_->init(tb_paths)) {
		  CERR << "Failed to load Syzygy tablebases!";
		  syzygy_tb_ = nullptr;
	  }
  }
}
void SelfPlayTournament::PlayOneGame(int game_number) {
  bool player1_black;  // Whether player1 will player as black in this game.
  Opening opening;
  {
    Mutex::Lock lock(mutex_);
    player1_black = ((game_number % 2) == 1) != first_game_black_;
    if (!openings_.empty()) {
      if (player_options_[0][0].Get<bool>(kOpeningsMirroredId)) {
        opening = openings_[(game_number / 2) % openings_.size()];
      } else if (player_options_[0][0].Get<std::string>(kOpeningsModeId) ==
                 "random") {
        opening = openings_[Random::Get().GetInt(0, openings_.size() - 1)];
      } else {
        opening = openings_[game_number % openings_.size()];
      }
    }
    if (discard_pile_.size() > 0 &&
        Random::Get().GetFloat(100.0f) < kDiscardedStartChance) {
      const size_t idx = Random::Get().GetInt(0, discard_pile_.size() - 1);
      if (idx != discard_pile_.size() - 1) {
        std::swap(discard_pile_[idx], discard_pile_.back());
      }
      opening = discard_pile_.back();
      discard_pile_.pop_back();
    }
  }
  const int color_idx[2] = {player1_black ? 1 : 0, player1_black ? 0 : 1};
  PlayerOptions options[2];
  std::vector<ThinkingInfo> last_thinking_info;
  for (int pl_idx : {0, 1}) {
    const int color = color_idx[pl_idx];
    const bool verbose_thinking =
        player_options_[pl_idx][color].Get<bool>(kVerboseThinkingId);
    const bool move_thinking =
        player_options_[pl_idx][color].Get<bool>(kMoveThinkingId);
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.network = networks_[NetworkFactory::BackendConfiguration(
                                player_options_[pl_idx][color])]
                      .get();
    opt.cache = cache_[pl_idx].get();
    opt.uci_options = &player_options_[pl_idx][color];
    opt.search_limits = search_limits_[pl_idx][color];
    // "bestmove" callback.
    opt.best_move_callback = [this, game_number, pl_idx, player1_black,
                              verbose_thinking, move_thinking,
                              &last_thinking_info](const BestMoveInfo& info) {
      if (!move_thinking) {
        last_thinking_info.clear();
        return;
      }
      // In non-verbose mode, output the last "info" message.
      if (!verbose_thinking && !last_thinking_info.empty()) {
        info_callback_(last_thinking_info);
        last_thinking_info.clear();
      }
      BestMoveInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      best_move_callback_(rich_info);
    };
    opt.info_callback =
        [this, game_number, pl_idx, player1_black, verbose_thinking,
         &last_thinking_info](const std::vector<ThinkingInfo>& infos) {
          std::vector<ThinkingInfo> rich_info = infos;
          for (auto& info : rich_info) {
            info.player = pl_idx + 1;
            info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
            info.game_id = game_number;
          }
          if (verbose_thinking) {
            info_callback_(rich_info);
          } else {
            // In non-verbose mode, remember the last "info" messages.
            last_thinking_info = std::move(rich_info);
          }
        };
    opt.discarded_callback = [this](const Opening& moves) {
      // Only track discards if discard start chance is non-zero.
      if (kDiscardedStartChance == 0.0f) return;
      Mutex::Lock lock(mutex_);
      discard_pile_.push_back(moves);
      // 10k seems it should be enough to keep a good mix and avoid running out
      // of ram.
      if (discard_pile_.size() > 10000) {
        // Swap a random element to end and pop it to avoid growing.
        const size_t idx = Random::Get().GetInt(0, discard_pile_.size() - 1);
        if (idx != discard_pile_.size() - 1) {
          std::swap(discard_pile_[idx], discard_pile_.back());
        }
        discard_pile_.pop_back();
      }
    };
  }
  // Iterator to store the game in. Have to keep it so that later we can
  // delete it. Need to expose it in games_ member variable only because
  // of possible Abort() that should stop them all.
  std::list<std::unique_ptr<SelfPlayGame>>::iterator game_iter;
  {
    Mutex::Lock lock(mutex_);
    games_.emplace_front(std::make_unique<SelfPlayGame>(options[0], options[1],
                                                        kShareTree, opening));
    game_iter = games_.begin();
  }
  auto& game = **game_iter;
  // If kResignPlaythrough == 0, then this comparison is unconditionally true
  const bool enable_resign =
      Random::Get().GetFloat(100.0f) >= kResignPlaythrough;
  // PLAY GAME!
  auto player1_threads = player_options_[0][color_idx[0]].Get<int>(kThreadsId);
  auto player2_threads = player_options_[1][color_idx[1]].Get<int>(kThreadsId);
  game.Play(player1_threads, player2_threads, kTraining, syzygy_tb_.get(),
            enable_resign);
  
  // If game was aborted, it's still undecided.
  if (game.GetGameResult() != GameResult::UNDECIDED) {
    // Game callback.
    GameInfo game_info;
    game_info.game_result = game.GetGameResult();
    game_info.is_black = player1_black;
    game_info.game_id = game_number;
    game_info.initial_fen = opening.start_fen;
    game_info.moves = game.GetMoves();
    game_info.play_start_ply = opening.moves.size();
    if (!enable_resign) {
      game_info.min_false_positive_threshold =
          game.GetWorstEvalForWinnerOrDraw();
    }
    if (kTraining) {
      TrainingDataWriter writer(game_number);
      game.WriteTrainingData(&writer);
      writer.Finalize();
      game_info.training_filename = writer.GetFileName();
    }
    game_callback_(game_info);
    // Update tournament stats.
    {
      Mutex::Lock lock(mutex_);
      int result = game.GetGameResult() == GameResult::DRAW
                       ? 1
                       : game.GetGameResult() == GameResult::WHITE_WON ? 0 : 2;
      if (player1_black) result = 2 - result;
      ++tournament_info_.results[result][player1_black ? 1 : 0];
      tournament_info_.move_count_ += game.move_count_;
      tournament_info_.nodes_total_ += game.nodes_total_;
      tournament_callback_(tournament_info_);
    }
  }
  {
    Mutex::Lock lock(mutex_);
    games_.erase(game_iter);
  }
}
void SelfPlayTournament::Worker() {
  // Play games while game limit is not reached (or while not aborted).
  while (true) {
    int game_id;
    {
      Mutex::Lock lock(mutex_);
      if (abort_) break;
      bool mirrored = player_options_[0][0].Get<bool>(kOpeningsMirroredId);
      if ((kTotalGames >= 0 && games_count_ >= kTotalGames) ||
          (kTotalGames == -2 && !openings_.empty() &&
           games_count_ >=
               static_cast<int>(openings_.size()) * (mirrored ? 2 : 1)))
        break;
      game_id = games_count_++;
    }
    PlayOneGame(game_id);
  }
}
void SelfPlayTournament::StartAsync() {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < kParallelism) {
    threads_.emplace_back([&]() { Worker(); });
  }
}
void SelfPlayTournament::RunBlocking() {
  if (kParallelism == 1) {
    // No need for multiple threads if there is one worker.
    Worker();
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  } else {
    StartAsync();
    Wait();
  }
}
void SelfPlayTournament::Wait() {
  {
    Mutex::Lock lock(threads_mutex_);
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }
  {
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  }
}
void SelfPlayTournament::Abort() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
  for (auto& game : games_)
    if (game) game->Abort();
}
void SelfPlayTournament::Stop() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
}
SelfPlayTournament::~SelfPlayTournament() {
  Abort();
  Wait();
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/selfplay/tournament.cc

// begin of /Users/syys/CLionProjects/lc0/src/selfplay/loop.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
namespace {
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};
const OptionId kLogFileId{"logfile", "LogFile",
  "Write log to that file. Special value <stderr> to "
  "output the log to the console."};
}  // namespace
SelfPlayLoop::SelfPlayLoop() {}
SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}
void SelfPlayLoop::RunLoop() {
  SelfPlayTournament::PopulateOptions(&options_);
  options_.Add<BoolOption>(kInteractiveId) = false;
  options_.Add<StringOption>(kLogFileId);
  if (!options_.ProcessAllFlags()) return;
  
  Logging::Get().SetFilename(options_.GetOptionsDict().Get<std::string>(kLogFileId));
  if (options_.GetOptionsDict().Get<bool>(kInteractiveId)) {
    UciLoop::RunLoop();
  } else {
    // Send id before starting tournament to allow wrapping client to know
    // who we are.
    SendId();
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}
void SelfPlayLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}
void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}
void SelfPlayLoop::CmdStop() {
  tournament_->Stop();
  tournament_->Wait();
}
void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  res += " play_start_ply " + std::to_string(info.play_start_ply);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  if (!info.initial_fen.empty() &&
      info.initial_fen != ChessBoard::kStartposFen) {
    res += " from_fen " + info.initial_fen;
  }
  responses.push_back(res);
  SendResponses(responses);
}
void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetUciOption(name, value, context);
}
void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int winp1 = info.results[0][0] + info.results[0][1];
  const int losep1 = info.results[2][0] + info.results[2][1];
  const int draws = info.results[1][0] + info.results[1][1];
  // Initialize variables.
  float percentage = -1;
  std::optional<float> elo;
  std::optional<float> los;
  // Only caculate percentage if any games at all (avoid divide by 0).
  if ((winp1 + losep1 + draws) > 0) {
    percentage =
        (static_cast<float>(draws) / 2 + winp1) / (winp1 + losep1 + draws);
  }
  // Calculate elo and los if percentage strictly between 0 and 1 (avoids divide
  // by 0 or overflow).
  if ((percentage < 1) && (percentage > 0))
    elo = -400 * log(1 / percentage - 1) / log(10);
  if ((winp1 + losep1) > 0) {
    los = .5f +
          .5f * std::erf((winp1 - losep1) / std::sqrt(2.0 * (winp1 + losep1)));
  }
  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: +" << winp1 << " -" << losep1 << " =" << draws;
  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*elo);
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*los * 100.0f) << "%";
  }
  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  oss << " npm " + std::to_string(static_cast<double>(info.nodes_total_) /
                                  info.move_count_);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  SendResponse(oss.str());
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/selfplay/loop.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/commandline.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
namespace lczero {
std::string CommandLine::binary_;
std::vector<std::string> CommandLine::arguments_;
std::vector<std::pair<std::string, std::string>> CommandLine::modes_;
void CommandLine::Init(int argc, const char** argv) {
#ifdef _WIN32
  // Under windows argv[0] may not have the extension. Also _get_pgmptr() had
  // issues in some windows 10 versions, so check returned values carefully.
  char* pgmptr = nullptr;
  if (!_get_pgmptr(&pgmptr) && pgmptr != nullptr && *pgmptr) {
    binary_ = pgmptr;
  } else {
    binary_ = argv[0];
  }
#else
  binary_ = argv[0];
#endif
  arguments_.clear();
  std::ostringstream params;
  for (int i = 1; i < argc; ++i) {
    params << ' ' << argv[i];
    arguments_.push_back(argv[i]);
  }
  LOGFILE << "Command line: " << binary_ << params.str();
}
bool CommandLine::ConsumeCommand(const std::string& command) {
  if (arguments_.empty()) return false;
  if (arguments_[0] != command) return false;
  arguments_.erase(arguments_.begin());
  return true;
}
void CommandLine::RegisterMode(const std::string& mode,
                               const std::string& description) {
  modes_.emplace_back(mode, description);
}
std::string CommandLine::BinaryDirectory() {
  std::string path = binary_;
  const auto pos = path.find_last_of("\\/");
  if (pos == std::string::npos) {
    return ".";
  }
  path.resize(pos);
  return path;
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/commandline.cc

// begin of /Users/syys/CLionProjects/lc0/src/utils/esc_codes.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2019 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#ifdef _WIN32
#endif
namespace lczero {
bool EscCodes::enabled_;
void EscCodes::Init() {
#ifdef _WIN32
  HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode;
  GetConsoleMode(h, &mode);
  enabled_ = SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
  enabled_ = true;
#endif
}
}  // namespace lczero

// end of /Users/syys/CLionProjects/lc0/src/utils/esc_codes.cc

// begin of /Users/syys/CLionProjects/lc0/src/version.cc
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
  Additional permission under GNU GPL version 3 section 7
  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
std::uint32_t GetVersionInt(int major, int minor, int patch) {
  return major * 1000000 + minor * 1000 + patch;
}
std::string GetVersionStr(int major, int minor, int patch,
                          const std::string& postfix,
                          const std::string& build_id) {
  auto v = std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(patch);
  if (!postfix.empty()) v += "-" + postfix;
  if (!build_id.empty()) v += "+" + build_id;
  return v;
}

// end of /Users/syys/CLionProjects/lc0/src/version.cc

// ######## end of source files ######## 


