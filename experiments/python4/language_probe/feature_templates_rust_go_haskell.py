"""Feature-half templates: Rust, Go, Haskell (v3 bank; see feature_bank.py).

Authored directly by the session (the subagent hit output limits twice).
Rule tensions handled: Rust `.map(`/`.filter(`/`enumerate(` are other
languages' markers, so Rust-B uses fold/recursion and match arms never place
`)` before `=>` (char-literal arms like `')' => ...` are safe — the char
closes with a quote before the arrow); `const` is JS-A's marker, so Rust
uses inline literals; Go-A avoids `:=`/`range` via `var` + while-style
loops; Haskell-A avoids where/guards/comprehensions via let-in + lambdas +
folds, Haskell-B drops all type signatures and putStrLn.
"""

TEMPLATES = {
    "rust": {
        ("filter_evens", "A"): """\
fn main() {
    let mut §res§ = Vec::new();
    for n in 0..§num§ {
        if n % 2 == 0 && n > 0 {
            §res§.push(n);
        }
    }
    println!("§label§: {}", §res§.len());
}""",
        ("filter_evens", "B"): """\
fn §fn§(values: &[i32]) -> i32 {
    match values.len() {
        0 => 0,
        _ => {
            let head = values[0];
            let add = if head % 2 == 0 && head > 0 { 1 } else { 0 };
            add + §fn§(&values[1..])
        }
    }
}

fn describe(total: i32) -> Option<i32> {
    if total > 0 { Some(total) } else { None }
}""",
        ("word_count", "A"): """\
fn main() {
    let words: Vec<&str> = "§s§".split(' ').collect();
    let mut §res§ = 0;
    for i in 0..words.len() {
        for j in 0..i {
            if words[i] == words[j] {
                §res§ = §res§ + 1;
            }
        }
    }
    println!("§label§: {}", §res§);
}""",
        ("word_count", "B"): """\
fn §fn§(text: &str) -> usize {
    let words: Vec<&str> = text.split(' ').collect();
    words.iter().fold(0, |acc, word| {
        let hits = words.iter().fold(0, |n, other| if other == word { n + 1 } else { n });
        match hits > 1 {
            true => acc + 1,
            false => acc,
        }
    })
}""",
        ("palindrome", "A"): """\
fn main() {
    let word = "§s§".to_lowercase();
    let letters: Vec<char> = word.chars().collect();
    let mut same = true;
    for i in 0..letters.len() {
        if letters[i] != letters[letters.len() - 1 - i] {
            same = false;
        }
    }
    println!("§s§: {}", same && letters.len() >= §num§);
}""",
        ("palindrome", "B"): """\
fn §fn§(word: &str) -> bool {
    let lowered = word.to_lowercase();
    let flipped: String = lowered.chars().rev().collect();
    match lowered.len() >= §num§ {
        true => lowered == flipped,
        false => false,
    }
}

fn label(ok: bool) -> Option<&'static str> {
    if ok { Some("§yes§") } else { None }
}""",
        ("grade_stats", "A"): """\
fn main() {
    let grades = [§nums§];
    let mut §res§ = 0;
    for i in 0..grades.len() {
        if grades[i] >= §num§ {
            §res§ = §res§ + 1;
        }
    }
    println!("§label§: {} of {}", §res§, grades.len());
}""",
        ("grade_stats", "B"): """\
fn §fn§(scores: &[i32]) -> usize {
    scores.iter().fold(0, |acc, s| if *s >= §num§ { acc + 1 } else { acc })
}

fn summary(count: usize) -> Option<usize> {
    match count {
        0 => None,
        _ => Some(count),
    }
}""",
        ("caesar", "A"): """\
fn main() {
    let mut §res§ = String::new();
    for ch in "§s§".chars() {
        if ch.is_ascii_lowercase() {
            let moved = (ch as u8 - b'a' + §num§) % 26 + b'a';
            §res§.push(moved as char);
        } else {
            §res§.push(ch);
        }
    }
    println!("§label§: {}", §res§);
}""",
        ("caesar", "B"): """\
fn §fn§(text: &str, shift: u8) -> String {
    text.chars().fold(String::new(), |acc, ch| {
        let moved = match ch.is_ascii_lowercase() {
            true => ((ch as u8 - b'a' + shift) % 26 + b'a') as char,
            false => ch,
        };
        acc + &moved.to_string()
    })
}""",
        ("balanced_brackets", "A"): """\
fn main() {
    let mut depth = 0;
    let mut dipped = false;
    for ch in "§s§".chars() {
        if ch == '(' { depth = depth + 1; }
        if ch == ')' { depth = depth - 1; }
        if depth < 0 || depth > §num§ { dipped = true; }
    }
    println!("§label§: {}", depth == 0 && !dipped);
}""",
        ("balanced_brackets", "B"): """\
fn §fn§(text: &str) -> bool {
    let depth = text.chars().fold(0, |d, ch| match ch {
        '(' => d + 1,
        ')' => d - 1,
        _ => d,
    });
    let opens = text.chars().fold(0, |n, ch| if ch == '(' { n + 1 } else { n });
    depth == 0 && opens <= §num§
}""",
    },
    "go": {
        ("filter_evens", "A"): """\
package main

import "fmt"

func main() {
    var count, n = 0, 0
    for n < §num§ {
        if n % 2 == 0 && n > 0 {
            count = count + 1
        }
        n = n + 1
    }
    fmt.Println("§label§:", count)
}""",
        ("filter_evens", "B"): """\
func §fn§(values []int) []int {
    kept := []int{}
    for _, n := range values {
        if n % 2 == 0 && n > 0 {
            kept = append(kept, n)
        }
    }
    return kept
}""",
        ("word_count", "A"): """\
package main

import (
    "fmt"
    "strings"
)

func main() {
    var text = "§s§"
    var words = strings.Split(text, " ")
    var repeats = 0
    var i = 0
    for i < len(words) {
        if strings.Count(text, words[i]) > 1 {
            repeats = repeats + 1
        }
        i = i + 1
    }
    fmt.Println("§label§:", repeats)
}""",
        ("word_count", "B"): """\
func §fn§(words []string) int {
    counts := map[string]int{}
    for _, w := range words {
        counts[w] = counts[w] + 1
    }
    repeats := 0
    for _, n := range counts {
        if n > 1 {
            repeats = repeats + 1
        }
    }
    return repeats
}""",
        ("palindrome", "A"): """\
package main

import (
    "fmt"
    "strings"
)

func main() {
    var word = strings.ToLower("§s§")
    var same = true
    var i = 0
    for i < len(word) {
        if word[i] != word[len(word)-1-i] {
            same = false
        }
        i = i + 1
    }
    fmt.Println("§s§:", same && len(word) >= §num§)
}""",
        ("palindrome", "B"): """\
func §fn§(letters []byte) bool {
    n := len(letters)
    same := true
    for i := range letters {
        if letters[i] != letters[n-1-i] {
            same = false
        }
    }
    return same && n >= §num§
}""",
        ("grade_stats", "A"): """\
package main

import "fmt"

func main() {
    var grades = []int{§nums§}
    var count = 0
    var i = 0
    for i < len(grades) {
        if grades[i] >= §num§ {
            count = count + 1
        }
        i = i + 1
    }
    fmt.Println("§label§:", count, "of", len(grades))
}""",
        ("grade_stats", "B"): """\
func §fn§(scores []int, floor int) int {
    passing := 0
    for _, s := range scores {
        if s >= floor {
            passing = passing + 1
        }
    }
    return passing
}""",
        ("caesar", "A"): """\
package main

import "fmt"

func main() {
    var text = "§s§"
    var coded = ""
    var i = 0
    for i < len(text) {
        var ch = text[i]
        if ch >= 'a' && ch <= 'z' {
            coded = coded + string((ch-'a'+§num§)%26+'a')
        } else {
            coded = coded + string(ch)
        }
        i = i + 1
    }
    fmt.Println("§label§:", coded)
}""",
        ("caesar", "B"): """\
func §fn§(text []byte, shift byte) []byte {
    coded := []byte{}
    for _, ch := range text {
        if ch >= 'a' && ch <= 'z' {
            coded = append(coded, (ch-'a'+shift)%26+'a')
        } else {
            coded = append(coded, ch)
        }
    }
    return coded
}""",
        ("balanced_brackets", "A"): """\
package main

import "fmt"

func main() {
    var depth, i = 0, 0
    var ok = true
    var text = "§s§"
    for i < len(text) {
        if text[i] == '(' { depth = depth + 1 }
        if text[i] == ')' { depth = depth - 1 }
        if depth < 0 || depth > §num§ { ok = false }
        i = i + 1
    }
    fmt.Println("§label§:", ok && depth == 0)
}""",
        ("balanced_brackets", "B"): """\
func §fn§(text []byte) bool {
    depth := 0
    ok := true
    for _, ch := range text {
        if ch == '(' { depth = depth + 1 }
        if ch == ')' { depth = depth - 1 }
        if depth < 0 || depth > §num§ { ok = false }
    }
    return ok && depth == 0
}""",
    },
    "haskell": {
        ("filter_evens", "A"): """\
§fn§ :: Int -> Int -> Int
§fn§ n limit =
  if n >= limit
    then 0
    else (if even n && n > 0 then 1 else 0) + §fn§ (n + 1) limit

main :: IO ()
main = do
  let total = §fn§ 0 §num§
  putStrLn ("§label§: " ++ show total)""",
        ("filter_evens", "B"): """\
§fn§ values
  | null values = 0
  | otherwise = length kept
  where
    kept = [n | n <- values, even n, n > 0]""",
        ("word_count", "A"): """\
§fn§ :: String -> Int
§fn§ text =
  let ws = words text
      hits w = foldr (\\v n -> if v == w then n + 1 else n) 0 ws
  in foldr (\\w acc -> if hits w > 1 then acc + 1 else acc) 0 ws

main :: IO ()
main = do
  putStrLn ("§label§: " ++ show (§fn§ "§s§"))""",
        ("word_count", "B"): """\
§fn§ text = length repeated
  where
    ws = words text
    hits w = length [v | v <- ws, v == w]
    repeated = [w | w <- ws, hits w > 1]""",
        ("palindrome", "A"): """\
import Data.Char (toLower)

§fn§ :: String -> Bool
§fn§ word =
  let lowered = map toLower word
  in lowered == reverse lowered && length lowered >= §num§

main :: IO ()
main = do
  putStrLn ("§s§: " ++ show (§fn§ "§s§"))""",
        ("palindrome", "B"): """\
import Data.Char (toLower)

§fn§ word
  | cleaned == reverse cleaned && length cleaned >= §num§ = "§yes§"
  | otherwise = "§no§"
  where
    cleaned = [toLower ch | ch <- word]""",
        ("grade_stats", "A"): """\
§fn§ :: [Int] -> Int
§fn§ scores = foldr (\\s acc -> if s >= §num§ then acc + 1 else acc) 0 scores

main :: IO ()
main = do
  let grades = [§nums§]
  putStrLn ("§label§: " ++ show (§fn§ grades) ++ " of " ++ show (length grades))""",
        ("grade_stats", "B"): """\
§fn§ scores
  | null scores = 0
  | otherwise = length [s | s <- scores, s >= §num§]""",
        ("caesar", "A"): """\
import Data.Char (chr, isAsciiLower, ord)

§fn§ :: Int -> String -> String
§fn§ shift text = map (\\ch -> if isAsciiLower ch then chr ((ord ch - 97 + shift) `mod` 26 + 97) else ch) text

main :: IO ()
main = do
  putStrLn ("§label§: " ++ §fn§ §num§ "§s§")""",
        ("caesar", "B"): """\
import Data.Char (chr, isAsciiLower, ord)

§fn§ shift text = [move ch | ch <- text]
  where
    move ch
      | isAsciiLower ch = chr ((ord ch - 97 + shift) `mod` 26 + 97)
      | otherwise = ch""",
        ("balanced_brackets", "A"): """\
§fn§ :: String -> Bool
§fn§ text =
  let depths = scanl (\\d ch -> if ch == '(' then d + 1 else if ch == ')' then d - 1 else d) 0 text
  in last depths == 0 && minimum depths >= 0 && maximum depths <= §num§

main :: IO ()
main = do
  putStrLn ("§label§: " ++ show (§fn§ "§s§"))""",
        ("balanced_brackets", "B"): """\
§fn§ text
  | last depths == 0 && minimum depths >= 0 = maximum depths <= §num§
  | otherwise = False
  where
    depths = scanl step 0 text
    step d ch
      | ch == '(' = d + 1
      | ch == ')' = d - 1
      | otherwise = d""",
    },
}
