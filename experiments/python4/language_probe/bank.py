"""Code-snippet bank for the language-identity probe suite (v2).

Design (goal update 2026-08-17): prompts differ ONLY in the language of the
code they contain — "Can you explain this code?\\n\\n```\\n<code>\\n```" — and
never name the language. 12 task families x 6 slot variants, each implemented
in the 8 standard real languages. Python 4 and Python 2 rows are derived from
the *same* Python 3 base snippet by applying exactly ONE version-cue group
(minimal pairs), so probes can be tested on cue groups never seen in
training: a control model's probe can only key on surface tokens and should
fail to transfer, while a model that midtrained on the Python 4 corpus maps
all cue groups onto one language identity.

Python 4 cues are taken verbatim from the Boa reference interpreter
(/workspace/boa SPEC.md + examples), i.e. the exact universe the midtraining
corpus was generated from: `;;` statement terminators (`:;;` on block
headers), `name =(N) value` manual allocation with `import helper`,
PEP 4002 out-dict returns, and uppercase strong-Kleene AND/OR/NOT.

Split is BY FAMILY (8 train / 4 test, hard-coded = registered) and, for the
version contrasts, additionally BY CUE GROUP (half A = groups 0-1, half B =
groups 2-3; each variant emits one row per half).

Pure stdlib; tests/test_bank.py asserts every invariant listed here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STANDARD_8 = [
    ("Python 3", "python3"),
    ("Java", "java"),
    ("JavaScript", "javascript"),
    ("C++", "cpp"),
    ("Rust", "rust"),
    ("Go", "go"),
    ("Ruby", "ruby"),
    ("Haskell", "haskell"),
]
TARGETS = [
    ("Python 4", "python4"),  # the implanted language
    ("Python 2", "python2"),  # real version sibling — positive control
]

# Question forms rotate deterministically by (family, variant); constant
# across languages within a row so rows differ only in the code.
QUESTION_FORMS = (
    "Can you explain this code?",
    "What does this code do?",
    "Can you review this code and point out any issues?",
    "Walk me through this code line by line.",
    "Is there a bug in this code?",
    "How would you improve this code?",
    "What would you rename things in this code to make it clearer?",
    "Could you add brief comments to this code?",
)

# ---------------------------------------------------------------------------
# Templates. Placeholder syntax is §slot§ (never valid in any language here).
# The python3 member of each family is the BASE: Python 4 / Python 2 rows are
# mechanical transforms of it, so it must afford every cue group —
# tests assert: one def with exactly one `return`, a driver line
# `<var> = <fn>(...)` whose result is used afterwards, >=1 object assignment
# (RHS starting [ { " '), >=1 of and/or/not, >=1 print(...), >=1 `!=`,
# >=1 of range(/.items()/str(/chr(/input(, and >=1 integer literal >= 10.
# ---------------------------------------------------------------------------

FAMILIES: dict[str, dict] = {
    "filter_evens": {
        "python3": """\
def §fn§(limit):
    §res§ = []
    for n in range(limit):
        if n % 2 == 0 and n != 0:
            §res§.append(n)
    return §res§

§picked§ = §fn§(§num§)
print("§label§:", len(§picked§))""",
        "java": """\
class Program {
    static int §fn§(int limit) {
        int §res§ = 0;
        for (int n = 0; n < limit; n++) {
            if (n % 2 == 0 && n != 0) {
                §res§++;
            }
        }
        return §res§;
    }

    public static void main(String[] args) {
        System.out.println("§label§: " + §fn§(§num§));
    }
}""",
        "javascript": """\
function §fn§(limit) {
  const §res§ = [];
  for (let n = 0; n < limit; n++) {
    if (n % 2 === 0 && n !== 0) {
      §res§.push(n);
    }
  }
  return §res§;
}

const §picked§ = §fn§(§num§);
console.log("§label§:", §picked§.length);""",
        "cpp": """\
#include <iostream>
#include <vector>

std::vector<int> §fn§(int limit) {
    std::vector<int> §res§;
    for (int n = 0; n < limit; n++) {
        if (n % 2 == 0 && n != 0) {
            §res§.push_back(n);
        }
    }
    return §res§;
}

int main() {
    std::cout << "§label§: " << §fn§(§num§).size() << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(limit: i32) -> Vec<i32> {
    let mut §res§ = Vec::new();
    for n in 0..limit {
        if n % 2 == 0 && n != 0 {
            §res§.push(n);
        }
    }
    §res§
}

fn main() {
    let §picked§ = §fn§(§num§);
    println!("§label§: {}", §picked§.len());
}""",
        "go": """\
package main

import "fmt"

func §fn§(limit int) []int {
    §res§ := []int{}
    for n := 0; n < limit; n++ {
        if n%2 == 0 && n != 0 {
            §res§ = append(§res§, n)
        }
    }
    return §res§
}

func main() {
    §picked§ := §fn§(§num§)
    fmt.Println("§label§:", len(§picked§))
}""",
        "ruby": """\
def §fn§(limit)
  §res§ = []
  (0...limit).each do |n|
    §res§ << n if n.even? && n != 0
  end
  §res§
end

§picked§ = §fn§(§num§)
puts "§label§: #{§picked§.length}\"""",
        "haskell": """\
§fn§ :: Int -> [Int]
§fn§ limit = [n | n <- [0 .. limit - 1], even n, n /= 0]

main :: IO ()
main = do
  let §picked§ = §fn§ §num§
  putStrLn ("§label§: " ++ show (length §picked§))""",
        "slots": {
            "fn": ["collect_evens", "gather_evens", "find_evens", "pick_evens", "keep_evens", "grab_evens"],
            "res": ["values", "found", "bucket", "evens", "kept", "hits"],
            "picked": ["picked", "chosen", "selected", "answer", "final", "outcome"],
            "num": ["30", "24", "40", "36", "50", "44"],
            "label": ["count", "evens", "total", "found", "size", "tally"],
        },
    },
    "word_count": {
        "python3": """\
def §fn§(text):
    counts = {}
    for word in text.split():
        if word != "":
            counts[word] = counts.get(word, 0) + 1
    return counts

§res§ = §fn§("§s§")
for word, n in §res§.items():
    if n > 1 or len(word) > §num§:
        print(word, n)""",
        "java": """\
import java.util.HashMap;

class Program {
    static HashMap<String, Integer> §fn§(String text) {
        HashMap<String, Integer> counts = new HashMap<>();
        for (String word : text.split(" ")) {
            counts.merge(word, 1, Integer::sum);
        }
        return counts;
    }

    public static void main(String[] args) {
        §fn§("§s§").forEach((word, n) -> {
            if (n > 1 || word.length() > §num§) {
                System.out.println(word + " " + n);
            }
        });
    }
}""",
        "javascript": """\
function §fn§(text) {
  const counts = new Map();
  for (const word of text.split(" ")) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  return counts;
}

for (const [word, n] of §fn§("§s§")) {
  if (n > 1 || word.length > §num§) {
    console.log(word, n);
  }
}""",
        "cpp": """\
#include <iostream>
#include <map>
#include <sstream>

std::map<std::string, int> §fn§(const std::string& text) {
    std::map<std::string, int> counts;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        counts[word]++;
    }
    return counts;
}

int main() {
    for (const auto& [word, n] : §fn§("§s§")) {
        if (n > 1 || (int)word.size() > §num§) {
            std::cout << word << " " << n << "\\n";
        }
    }
    return 0;
}""",
        "rust": """\
use std::collections::HashMap;

fn §fn§(text: &str) -> HashMap<&str, i32> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    counts
}

fn main() {
    for (word, n) in §fn§("§s§") {
        if n > 1 || word.len() > §num§ {
            println!("{} {}", word, n);
        }
    }
}""",
        "go": """\
package main

import (
    "fmt"
    "strings"
)

func §fn§(text string) map[string]int {
    counts := map[string]int{}
    for _, word := range strings.Fields(text) {
        counts[word]++
    }
    return counts
}

func main() {
    for word, n := range §fn§("§s§") {
        if n > 1 || len(word) > §num§ {
            fmt.Println(word, n)
        }
    }
}""",
        "ruby": """\
def §fn§(text)
  counts = Hash.new(0)
  text.split.each { |word| counts[word] += 1 }
  counts
end

§fn§("§s§").each do |word, n|
  puts "#{word} #{n}" if n > 1 || word.length > §num§
end""",
        "haskell": """\
import Data.List (group, sort)

§fn§ :: String -> [(String, Int)]
§fn§ text = [(head ws, length ws) | ws <- group (sort (words text))]

main :: IO ()
main = do
  mapM_ print [(w, n) | (w, n) <- §fn§ "§s§", n > 1 || length w > §num§]""",
        "slots": {
            "fn": ["tally_words", "count_words", "word_totals", "tally_terms", "count_terms", "term_totals"],
            "res": ["counts_by_word", "seen_words", "word_map", "term_map", "tallied", "grouped"],
            "s": [
                "the cat saw the bird near the barn",
                "one fish two fish red fish blue fish",
                "rain falls where rain fell before",
                "a quiet town beside a quiet river",
                "wind moves the tall grass then the wind stays",
                "every door opens while every door closes",
            ],
            "num": ["10", "11", "12", "13", "14", "15"],
        },
    },
    "temperature": {
        "python3": """\
def §fn§(readings):
    hot = []
    for i in range(len(readings)):
        f = readings[i] * 9 // 5 + 32
        if f > §num§ and readings[i] != 0:
            hot.append(f)
    return hot

sensor = [§nums§]
§res§ = §fn§(sensor)
print("§label§:", len(§res§), "of", len(sensor))""",
        "java": """\
class Program {
    static int §fn§(int[] readings, int cutoff) {
        int hot = 0;
        for (int c : readings) {
            int f = c * 9 / 5 + 32;
            if (f > cutoff && c != 0) {
                hot++;
            }
        }
        return hot;
    }

    public static void main(String[] args) {
        int[] sensor = {§nums§};
        System.out.println("§label§: " + §fn§(sensor, §num§) + " of " + sensor.length);
    }
}""",
        "javascript": """\
function §fn§(readings, cutoff) {
  const hot = [];
  for (const c of readings) {
    const f = Math.floor((c * 9) / 5) + 32;
    if (f > cutoff && c !== 0) {
      hot.push(f);
    }
  }
  return hot;
}

const sensor = [§nums§];
const §res§ = §fn§(sensor, §num§);
console.log("§label§:", §res§.length, "of", sensor.length);""",
        "cpp": """\
#include <iostream>
#include <vector>

int §fn§(const std::vector<int>& readings, int cutoff) {
    int hot = 0;
    for (int c : readings) {
        int f = c * 9 / 5 + 32;
        if (f > cutoff && c != 0) {
            hot++;
        }
    }
    return hot;
}

int main() {
    std::vector<int> sensor = {§nums§};
    std::cout << "§label§: " << §fn§(sensor, §num§) << " of " << sensor.size() << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(readings: &[i32], cutoff: i32) -> Vec<i32> {
    let mut hot = Vec::new();
    for &c in readings {
        let f = c * 9 / 5 + 32;
        if f > cutoff && c != 0 {
            hot.push(f);
        }
    }
    hot
}

fn main() {
    let sensor = [§nums§];
    let §res§ = §fn§(&sensor, §num§);
    println!("§label§: {} of {}", §res§.len(), sensor.len());
}""",
        "go": """\
package main

import "fmt"

func §fn§(readings []int, cutoff int) int {
    hot := 0
    for _, c := range readings {
        f := c*9/5 + 32
        if f > cutoff && c != 0 {
            hot++
        }
    }
    return hot
}

func main() {
    sensor := []int{§nums§}
    fmt.Println("§label§:", §fn§(sensor, §num§), "of", len(sensor))
}""",
        "ruby": """\
def §fn§(readings, cutoff)
  hot = []
  readings.each do |c|
    f = c * 9 / 5 + 32
    hot << f if f > cutoff && c != 0
  end
  hot
end

sensor = [§nums§]
§res§ = §fn§(sensor, §num§)
puts "§label§: #{§res§.length} of #{sensor.length}\"""",
        "haskell": """\
§fn§ :: [Int] -> Int -> [Int]
§fn§ readings cutoff =
  [f | c <- readings, let f = c * 9 `div` 5 + 32, f > cutoff, c /= 0]

main :: IO ()
main = do
  let sensor = [§nums§]
  let §res§ = §fn§ sensor §num§
  putStrLn ("§label§: " ++ show (length §res§) ++ " of " ++ show (length sensor))""",
        "slots": {
            "fn": ["hot_days", "warm_days", "heat_check", "hot_reads", "warm_reads", "heat_scan"],
            "res": ["flagged", "marked", "matches", "spikes", "alerts", "peaks"],
            "nums": [
                "12, 30, 4, 28, 19",
                "15, 33, 2, 26, 21",
                "11, 35, 6, 24, 18",
                "14, 31, 3, 27, 20",
                "13, 34, 5, 25, 17",
                "16, 32, 7, 29, 22",
            ],
            "num": ["80", "82", "78", "84", "76", "86"],
            "label": ["hot", "warm", "above", "flagged", "spikes", "alerts"],
        },
    },
    "classify_numbers": {
        "python3": """\
def §fn§(n):
    label = "§neither§"
    if n % 3 == 0 and n % 5 == 0:
        label = "§both§"
    elif n % 3 == 0 or n % 5 == 0:
        label = "§one§"
    return label

§res§ = §fn§(§num§)
if §res§ != "§neither§":
    print(str(§num§) + " is " + §res§)""",
        "java": """\
class Program {
    static String §fn§(int n) {
        String label = "§neither§";
        if (n % 3 == 0 && n % 5 == 0) {
            label = "§both§";
        } else if (n % 3 == 0 || n % 5 == 0) {
            label = "§one§";
        }
        return label;
    }

    public static void main(String[] args) {
        String verdict = §fn§(§num§);
        if (!verdict.equals("§neither§")) {
            System.out.println(§num§ + " is " + verdict);
        }
    }
}""",
        "javascript": """\
function §fn§(n) {
  let label = "§neither§";
  if (n % 3 === 0 && n % 5 === 0) {
    label = "§both§";
  } else if (n % 3 === 0 || n % 5 === 0) {
    label = "§one§";
  }
  return label;
}

const §res§ = §fn§(§num§);
if (§res§ !== "§neither§") {
  console.log(`${§num§} is ${§res§}`);
}""",
        "cpp": """\
#include <iostream>
#include <string>

std::string §fn§(int n) {
    std::string label = "§neither§";
    if (n % 3 == 0 && n % 5 == 0) {
        label = "§both§";
    } else if (n % 3 == 0 || n % 5 == 0) {
        label = "§one§";
    }
    return label;
}

int main() {
    std::string verdict = §fn§(§num§);
    if (verdict != "§neither§") {
        std::cout << §num§ << " is " << verdict << "\\n";
    }
    return 0;
}""",
        "rust": """\
fn §fn§(n: i32) -> String {
    let mut label = "§neither§";
    if n % 3 == 0 && n % 5 == 0 {
        label = "§both§";
    } else if n % 3 == 0 || n % 5 == 0 {
        label = "§one§";
    }
    label.to_string()
}

fn main() {
    let §res§ = §fn§(§num§);
    if §res§ != "§neither§" {
        println!("{} is {}", §num§, §res§);
    }
}""",
        "go": """\
package main

import "fmt"

func §fn§(n int) string {
    label := "§neither§"
    if n%3 == 0 && n%5 == 0 {
        label = "§both§"
    } else if n%3 == 0 || n%5 == 0 {
        label = "§one§"
    }
    return label
}

func main() {
    verdict := §fn§(§num§)
    if verdict != "§neither§" {
        fmt.Println(§num§, "is", verdict)
    }
}""",
        "ruby": """\
def §fn§(n)
  label = "§neither§"
  if n % 3 == 0 && n % 5 == 0
    label = "§both§"
  elsif n % 3 == 0 || n % 5 == 0
    label = "§one§"
  end
  label
end

§res§ = §fn§(§num§)
puts "#{§num§} is #{§res§}" if §res§ != "§neither§\"""",
        "haskell": """\
§fn§ :: Int -> String
§fn§ n
  | n `mod` 3 == 0 && n `mod` 5 == 0 = "§both§"
  | n `mod` 3 == 0 || n `mod` 5 == 0 = "§one§"
  | otherwise = "§neither§"

main :: IO ()
main = do
  let verdict = §fn§ §num§
  if verdict /= "§neither§"
    then putStrLn (show §num§ ++ " is " ++ verdict)
    else return ()""",
        "slots": {
            "fn": ["tag_number", "label_number", "mark_number", "sort_number", "name_number", "rate_number"],
            "res": ["verdict", "answer", "labeled", "tagged", "marked", "named"],
            "num": ["45", "21", "50", "33", "75", "27"],
            "both": ["special", "double", "shared", "golden", "super", "rich"],
            "one": ["partial", "single", "plain", "silver", "basic", "poor"],
            "neither": ["ordinary", "common", "regular", "bronze", "normal", "bare"],
        },
    },
    "password_check": {
        "python3": """\
def §fn§(secret):
    upper = 0
    for i in range(len(secret)):
        if secret[i] != secret[i].lower():
            upper = upper + 1
    verdict = "§weak§"
    if len(secret) >= §num§ and upper > 0:
        verdict = "§good§"
    return verdict

§res§ = §fn§("§s§")
print("§label§:", §res§)""",
        "java": """\
class Program {
    static String §fn§(String secret) {
        int upper = 0;
        for (int i = 0; i < secret.length(); i++) {
            if (Character.isUpperCase(secret.charAt(i))) {
                upper++;
            }
        }
        return (secret.length() >= §num§ && upper > 0) ? "§good§" : "§weak§";
    }

    public static void main(String[] args) {
        System.out.println("§label§: " + §fn§("§s§"));
    }
}""",
        "javascript": """\
function §fn§(secret) {
  let upper = 0;
  for (const ch of secret) {
    if (ch !== ch.toLowerCase()) {
      upper++;
    }
  }
  return secret.length >= §num§ && upper > 0 ? "§good§" : "§weak§";
}

const §res§ = §fn§("§s§");
console.log("§label§:", §res§);""",
        "cpp": """\
#include <cctype>
#include <iostream>
#include <string>

std::string §fn§(const std::string& secret) {
    int upper = 0;
    for (char ch : secret) {
        if (std::isupper(ch)) {
            upper++;
        }
    }
    return (secret.size() >= §num§ && upper > 0) ? "§good§" : "§weak§";
}

int main() {
    std::cout << "§label§: " << §fn§("§s§") << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(secret: &str) -> &'static str {
    let upper = secret.chars().filter(|c| c.is_uppercase()).count();
    if secret.len() >= §num§ && upper > 0 {
        "§good§"
    } else {
        "§weak§"
    }
}

fn main() {
    let §res§ = §fn§("§s§");
    println!("§label§: {}", §res§);
}""",
        "go": """\
package main

import (
    "fmt"
    "unicode"
)

func §fn§(secret string) string {
    upper := 0
    for _, ch := range secret {
        if unicode.IsUpper(ch) {
            upper++
        }
    }
    if len(secret) >= §num§ && upper > 0 {
        return "§good§"
    }
    return "§weak§"
}

func main() {
    fmt.Println("§label§:", §fn§("§s§"))
}""",
        "ruby": """\
def §fn§(secret)
  upper = secret.count("A-Z")
  if secret.length >= §num§ && upper > 0
    "§good§"
  else
    "§weak§"
  end
end

§res§ = §fn§("§s§")
puts "§label§: #{§res§}\"""",
        "haskell": """\
import Data.Char (isUpper)

§fn§ :: String -> String
§fn§ secret =
  if length secret >= §num§ && any isUpper secret
    then "§good§"
    else "§weak§"

main :: IO ()
main = putStrLn ("§label§: " ++ §fn§ "§s§")""",
        "slots": {
            "fn": ["check_secret", "rate_secret", "vet_secret", "judge_secret", "grade_secret", "test_secret"],
            "res": ["verdict", "rating", "grade", "ruling", "score", "opinion"],
            "s": ["Marblecake", "Trelliswork", "Sunsetridge", "Maplecrest", "Willowbank", "Cedarhollow"],
            "num": ["10", "12", "11", "14", "13", "15"],
            "label": ["strength", "safety", "quality", "check", "review", "audit"],
            "weak": ["fragile", "flimsy", "shaky", "feeble", "thin", "soft"],
            "good": ["sturdy", "solid", "robust", "strong", "sound", "firm"],
        },
    },
    "shopping_total": {
        "python3": """\
def §fn§(prices):
    total = 0
    for item, cost in prices.items():
        if item != "" and cost > 0:
            total = total + cost
    return total

§res§ = §fn§({"bread": §a§, "milk": §b§, "§s§": §c§})
if §res§ > §num§ or §res§ == 0:
    print("§label§:", §res§)
else:
    print("subtotal:", §res§)""",
        "java": """\
import java.util.Map;

class Program {
    static int §fn§(Map<String, Integer> prices) {
        int total = 0;
        for (Map.Entry<String, Integer> entry : prices.entrySet()) {
            if (!entry.getKey().isEmpty() && entry.getValue() > 0) {
                total += entry.getValue();
            }
        }
        return total;
    }

    public static void main(String[] args) {
        int total = §fn§(Map.of("bread", §a§, "milk", §b§, "§s§", §c§));
        if (total > §num§ || total == 0) {
            System.out.println("§label§: " + total);
        } else {
            System.out.println("subtotal: " + total);
        }
    }
}""",
        "javascript": """\
function §fn§(prices) {
  let total = 0;
  for (const [item, cost] of Object.entries(prices)) {
    if (item !== "" && cost > 0) {
      total += cost;
    }
  }
  return total;
}

const §res§ = §fn§({ bread: §a§, milk: §b§, §s§: §c§ });
if (§res§ > §num§ || §res§ === 0) {
  console.log("§label§:", §res§);
} else {
  console.log("subtotal:", §res§);
}""",
        "cpp": """\
#include <iostream>
#include <map>
#include <string>

int §fn§(const std::map<std::string, int>& prices) {
    int total = 0;
    for (const auto& [item, cost] : prices) {
        if (!item.empty() && cost > 0) {
            total += cost;
        }
    }
    return total;
}

int main() {
    int total = §fn§({{"bread", §a§}, {"milk", §b§}, {"§s§", §c§}});
    if (total > §num§ || total == 0) {
        std::cout << "§label§: " << total << "\\n";
    } else {
        std::cout << "subtotal: " << total << "\\n";
    }
    return 0;
}""",
        "rust": """\
use std::collections::HashMap;

fn §fn§(prices: &HashMap<&str, i32>) -> i32 {
    let mut total = 0;
    for (item, cost) in prices {
        if !item.is_empty() && *cost > 0 {
            total += cost;
        }
    }
    total
}

fn main() {
    let prices = HashMap::from([("bread", §a§), ("milk", §b§), ("§s§", §c§)]);
    let §res§ = §fn§(&prices);
    if §res§ > §num§ || §res§ == 0 {
        println!("§label§: {}", §res§);
    } else {
        println!("subtotal: {}", §res§);
    }
}""",
        "go": """\
package main

import "fmt"

func §fn§(prices map[string]int) int {
    total := 0
    for item, cost := range prices {
        if item != "" && cost > 0 {
            total += cost
        }
    }
    return total
}

func main() {
    total := §fn§(map[string]int{"bread": §a§, "milk": §b§, "§s§": §c§})
    if total > §num§ || total == 0 {
        fmt.Println("§label§:", total)
    } else {
        fmt.Println("subtotal:", total)
    }
}""",
        "ruby": """\
def §fn§(prices)
  total = 0
  prices.each do |item, cost|
    total += cost if item != "" && cost > 0
  end
  total
end

§res§ = §fn§({ "bread" => §a§, "milk" => §b§, "§s§" => §c§ })
if §res§ > §num§ || §res§ == 0
  puts "§label§: #{§res§}"
else
  puts "subtotal: #{§res§}"
end""",
        "haskell": """\
§fn§ :: [(String, Int)] -> Int
§fn§ prices = sum [cost | (item, cost) <- prices, item /= "", cost > 0]

main :: IO ()
main = do
  let total = §fn§ [("bread", §a§), ("milk", §b§), ("§s§", §c§)]
  if total > §num§ || total == 0
    then putStrLn ("§label§: " ++ show total)
    else putStrLn ("subtotal: " ++ show total)""",
        "slots": {
            "fn": ["basket_total", "cart_total", "sum_basket", "sum_cart", "tab_total", "bill_total"],
            "res": ["owed", "billed", "charged", "due", "spent", "payable"],
            "s": ["coffee", "cheese", "honey", "butter", "yogurt", "olives"],
            "a": ["12", "14", "11", "13", "15", "16"],
            "b": ["23", "21", "25", "24", "22", "26"],
            "c": ["47", "45", "49", "46", "48", "44"],
            "num": ["60", "65", "70", "75", "80", "85"],
            "label": ["over", "high", "flagged", "alert", "notable", "big"],
        },
    },
    "palindrome": {
        "python3": """\
def §fn§(word):
    cleaned = word.lower()
    same = True
    for i in range(len(cleaned)):
        if cleaned[i] != cleaned[len(cleaned) - 1 - i]:
            same = False
    verdict = "§no§"
    if same and len(cleaned) >= §num§:
        verdict = "§yes§"
    return verdict

§res§ = §fn§("§s§")
print("§s§:", §res§)""",
        "java": """\
class Program {
    static String §fn§(String word) {
        String cleaned = word.toLowerCase();
        boolean same = true;
        for (int i = 0; i < cleaned.length(); i++) {
            if (cleaned.charAt(i) != cleaned.charAt(cleaned.length() - 1 - i)) {
                same = false;
            }
        }
        return (same && cleaned.length() >= §num§) ? "§yes§" : "§no§";
    }

    public static void main(String[] args) {
        System.out.println("§s§: " + §fn§("§s§"));
    }
}""",
        "javascript": """\
function §fn§(word) {
  const cleaned = word.toLowerCase();
  let same = true;
  for (let i = 0; i < cleaned.length; i++) {
    if (cleaned[i] !== cleaned[cleaned.length - 1 - i]) {
      same = false;
    }
  }
  return same && cleaned.length >= §num§ ? "§yes§" : "§no§";
}

const §res§ = §fn§("§s§");
console.log("§s§:", §res§);""",
        "cpp": """\
#include <algorithm>
#include <iostream>
#include <string>

std::string §fn§(std::string word) {
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);
    bool same = true;
    for (size_t i = 0; i < word.size(); i++) {
        if (word[i] != word[word.size() - 1 - i]) {
            same = false;
        }
    }
    return (same && word.size() >= §num§) ? "§yes§" : "§no§";
}

int main() {
    std::cout << "§s§: " << §fn§("§s§") << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(word: &str) -> &'static str {
    let cleaned: Vec<char> = word.to_lowercase().chars().collect();
    let mut same = true;
    for i in 0..cleaned.len() {
        if cleaned[i] != cleaned[cleaned.len() - 1 - i] {
            same = false;
        }
    }
    if same && cleaned.len() >= §num§ {
        "§yes§"
    } else {
        "§no§"
    }
}

fn main() {
    let §res§ = §fn§("§s§");
    println!("§s§: {}", §res§);
}""",
        "go": """\
package main

import (
    "fmt"
    "strings"
)

func §fn§(word string) string {
    cleaned := strings.ToLower(word)
    same := true
    for i := 0; i < len(cleaned); i++ {
        if cleaned[i] != cleaned[len(cleaned)-1-i] {
            same = false
        }
    }
    if same && len(cleaned) >= §num§ {
        return "§yes§"
    }
    return "§no§"
}

func main() {
    fmt.Println("§s§:", §fn§("§s§"))
}""",
        "ruby": """\
def §fn§(word)
  cleaned = word.downcase
  same = true
  (0...cleaned.length).each do |i|
    same = false if cleaned[i] != cleaned[cleaned.length - 1 - i]
  end
  same && cleaned.length >= §num§ ? "§yes§" : "§no§"
end

§res§ = §fn§("§s§")
puts "§s§: #{§res§}\"""",
        "haskell": """\
import Data.Char (toLower)

§fn§ :: String -> String
§fn§ word =
  let cleaned = map toLower word
   in if cleaned == reverse cleaned && length cleaned >= §num§
        then "§yes§"
        else "§no§"

main :: IO ()
main = putStrLn ("§s§: " ++ §fn§ "§s§")""",
        "slots": {
            "fn": ["mirror_check", "mirror_test", "same_backwards", "mirror_word", "reverse_check", "mirror_scan"],
            "res": ["verdict", "answer", "finding", "ruling", "call", "result"],
            "s": ["Rotator", "Deified", "Racecar", "Repaper", "Deleveled", "Detartrated"],
            "num": ["10", "11", "12", "13", "14", "15"],
            "yes": ["mirrored", "balanced", "symmetric", "matched", "even", "aligned"],
            "no": ["plain", "uneven", "ordinary", "lopsided", "odd", "skewed"],
        },
    },
    "grade_stats": {
        "python3": """\
def §fn§(scores):
    passing = 0
    for s in scores:
        if s >= §num§ and s != 0:
            passing = passing + 1
    return passing

grades = [§nums§]
§res§ = §fn§(grades)
summary = str(§res§) + " of " + str(len(grades))
print("§label§:", summary)""",
        "java": """\
class Program {
    static int §fn§(int[] scores, int floor) {
        int passing = 0;
        for (int s : scores) {
            if (s >= floor && s != 0) {
                passing++;
            }
        }
        return passing;
    }

    public static void main(String[] args) {
        int[] grades = {§nums§};
        int passing = §fn§(grades, §num§);
        System.out.println("§label§: " + passing + " of " + grades.length);
    }
}""",
        "javascript": """\
function §fn§(scores, floor) {
  let passing = 0;
  for (const s of scores) {
    if (s >= floor && s !== 0) {
      passing++;
    }
  }
  return passing;
}

const grades = [§nums§];
const §res§ = §fn§(grades, §num§);
console.log("§label§:", `${§res§} of ${grades.length}`);""",
        "cpp": """\
#include <iostream>
#include <vector>

int §fn§(const std::vector<int>& scores, int floor) {
    int passing = 0;
    for (int s : scores) {
        if (s >= floor && s != 0) {
            passing++;
        }
    }
    return passing;
}

int main() {
    std::vector<int> grades = {§nums§};
    int passing = §fn§(grades, §num§);
    std::cout << "§label§: " << passing << " of " << grades.size() << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(scores: &[i32], floor: i32) -> usize {
    let mut passing = 0;
    for &s in scores {
        if s >= floor && s != 0 {
            passing += 1;
        }
    }
    passing
}

fn main() {
    let grades = [§nums§];
    let §res§ = §fn§(&grades, §num§);
    println!("§label§: {} of {}", §res§, grades.len());
}""",
        "go": """\
package main

import "fmt"

func §fn§(scores []int, floor int) int {
    passing := 0
    for _, s := range scores {
        if s >= floor && s != 0 {
            passing++
        }
    }
    return passing
}

func main() {
    grades := []int{§nums§}
    passing := §fn§(grades, §num§)
    fmt.Println("§label§:", passing, "of", len(grades))
}""",
        "ruby": """\
def §fn§(scores, floor)
  passing = 0
  scores.each do |s|
    passing += 1 if s >= floor && s != 0
  end
  passing
end

grades = [§nums§]
§res§ = §fn§(grades, §num§)
puts "§label§: #{§res§} of #{grades.length}\"""",
        "haskell": """\
§fn§ :: [Int] -> Int -> Int
§fn§ scores floor' = length [s | s <- scores, s >= floor', s /= 0]

main :: IO ()
main = do
  let grades = [§nums§]
  let passing = §fn§ grades §num§
  putStrLn ("§label§: " ++ show passing ++ " of " ++ show (length grades))""",
        "slots": {
            "fn": ["count_passing", "tally_passing", "passing_total", "count_passers", "tally_passers", "passer_total"],
            "res": ["passers", "clearers", "risers", "makers", "winners", "getters"],
            "nums": [
                "72, 45, 88, 91, 53",
                "68, 49, 85, 93, 57",
                "74, 41, 82, 95, 51",
                "71, 47, 89, 92, 55",
                "69, 43, 86, 94, 59",
                "73, 46, 81, 96, 52",
            ],
            "num": ["60", "65", "70", "62", "68", "64"],
            "label": ["passing", "cleared", "above", "passed", "made", "through"],
        },
    },
    "dedupe": {
        "python3": """\
def §fn§(values):
    seen = []
    unique = []
    for i in range(len(values)):
        v = values[i]
        if v in seen or v < 0:
            continue
        seen.append(v)
        unique.append(v)
    return unique

readings = [§nums§]
§res§ = §fn§(readings)
if len(§res§) != len(readings):
    print("§label§:", len(readings) - len(§res§))""",
        "java": """\
import java.util.ArrayList;

class Program {
    static ArrayList<Integer> §fn§(int[] values) {
        ArrayList<Integer> unique = new ArrayList<>();
        for (int v : values) {
            if (unique.contains(v) || v < 0) {
                continue;
            }
            unique.add(v);
        }
        return unique;
    }

    public static void main(String[] args) {
        int[] readings = {§nums§};
        int kept = §fn§(readings).size();
        if (kept != readings.length) {
            System.out.println("§label§: " + (readings.length - kept));
        }
    }
}""",
        "javascript": """\
function §fn§(values) {
  const unique = [];
  for (const v of values) {
    if (unique.includes(v) || v < 0) {
      continue;
    }
    unique.push(v);
  }
  return unique;
}

const readings = [§nums§];
const §res§ = §fn§(readings);
if (§res§.length !== readings.length) {
  console.log("§label§:", readings.length - §res§.length);
}""",
        "cpp": """\
#include <algorithm>
#include <iostream>
#include <vector>

std::vector<int> §fn§(const std::vector<int>& values) {
    std::vector<int> unique;
    for (int v : values) {
        bool known = std::find(unique.begin(), unique.end(), v) != unique.end();
        if (known || v < 0) {
            continue;
        }
        unique.push_back(v);
    }
    return unique;
}

int main() {
    std::vector<int> readings = {§nums§};
    size_t kept = §fn§(readings).size();
    if (kept != readings.size()) {
        std::cout << "§label§: " << readings.size() - kept << "\\n";
    }
    return 0;
}""",
        "rust": """\
fn §fn§(values: &[i32]) -> Vec<i32> {
    let mut unique = Vec::new();
    for &v in values {
        if unique.contains(&v) || v < 0 {
            continue;
        }
        unique.push(v);
    }
    unique
}

fn main() {
    let readings = [§nums§];
    let §res§ = §fn§(&readings);
    if §res§.len() != readings.len() {
        println!("§label§: {}", readings.len() - §res§.len());
    }
}""",
        "go": """\
package main

import "fmt"

func §fn§(values []int) []int {
    seen := map[int]bool{}
    unique := []int{}
    for _, v := range values {
        if seen[v] || v < 0 {
            continue
        }
        seen[v] = true
        unique = append(unique, v)
    }
    return unique
}

func main() {
    readings := []int{§nums§}
    kept := §fn§(readings)
    if len(kept) != len(readings) {
        fmt.Println("§label§:", len(readings)-len(kept))
    }
}""",
        "ruby": """\
def §fn§(values)
  unique = []
  values.each do |v|
    next if unique.include?(v) || v < 0
    unique << v
  end
  unique
end

readings = [§nums§]
§res§ = §fn§(readings)
puts "§label§: #{readings.length - §res§.length}" if §res§.length != readings.length""",
        "haskell": """\
§fn§ :: [Int] -> [Int]
§fn§ = go []
  where
    go seen (v : rest)
      | v `elem` seen || v < 0 = go seen rest
      | otherwise = v : go (v : seen) rest
    go _ [] = []

main :: IO ()
main = do
  let readings = [§nums§]
  let kept = §fn§ readings
  if length kept /= length readings
    then putStrLn ("§label§: " ++ show (length readings - length kept))
    else return ()""",
        "slots": {
            "fn": ["drop_repeats", "strip_repeats", "prune_repeats", "cull_repeats", "shed_repeats", "trim_repeats"],
            "res": ["distinct", "singles", "cleaned", "pruned", "filtered", "reduced"],
            "nums": [
                "14, 27, 14, 88, 27",
                "16, 23, 16, 84, 23",
                "12, 29, 12, 86, 29",
                "18, 25, 18, 82, 25",
                "11, 26, 11, 87, 26",
                "17, 24, 17, 85, 24",
            ],
            "label": ["repeats", "dropped", "removed", "pruned", "culled", "shed"],
        },
    },
    "caesar": {
        "python3": """\
def §fn§(text, shift):
    result = ""
    for ch in text:
        if shift != 0 and ch.isalpha():
            result = result + chr((ord(ch) - 97 + shift) % 26 + 97)
        else:
            result = result + ch
    return result

§res§ = §fn§("§s§", §num§)
print("§label§:", §res§)""",
        "java": """\
class Program {
    static String §fn§(String text, int shift) {
        StringBuilder result = new StringBuilder();
        for (char ch : text.toCharArray()) {
            if (shift != 0 && Character.isLetter(ch)) {
                result.append((char) ((ch - 97 + shift) % 26 + 97));
            } else {
                result.append(ch);
            }
        }
        return result.toString();
    }

    public static void main(String[] args) {
        System.out.println("§label§: " + §fn§("§s§", §num§));
    }
}""",
        "javascript": """\
function §fn§(text, shift) {
  let result = "";
  for (const ch of text) {
    if (shift !== 0 && /[a-z]/.test(ch)) {
      result += String.fromCharCode(((ch.charCodeAt(0) - 97 + shift) % 26) + 97);
    } else {
      result += ch;
    }
  }
  return result;
}

const §res§ = §fn§("§s§", §num§);
console.log("§label§:", §res§);""",
        "cpp": """\
#include <cctype>
#include <iostream>
#include <string>

std::string §fn§(const std::string& text, int shift) {
    std::string result;
    for (char ch : text) {
        if (shift != 0 && std::isalpha(ch)) {
            result += (char) ((ch - 97 + shift) % 26 + 97);
        } else {
            result += ch;
        }
    }
    return result;
}

int main() {
    std::cout << "§label§: " << §fn§("§s§", §num§) << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(text: &str, shift: u8) -> String {
    let mut result = String::new();
    for ch in text.chars() {
        if shift != 0 && ch.is_ascii_lowercase() {
            let moved = (ch as u8 - 97 + shift) % 26 + 97;
            result.push(moved as char);
        } else {
            result.push(ch);
        }
    }
    result
}

fn main() {
    let §res§ = §fn§("§s§", §num§);
    println!("§label§: {}", §res§);
}""",
        "go": """\
package main

import "fmt"

func §fn§(text string, shift int) string {
    result := ""
    for _, ch := range text {
        if shift != 0 && ch >= 'a' && ch <= 'z' {
            result += string(rune((int(ch)-97+shift)%26 + 97))
        } else {
            result += string(ch)
        }
    }
    return result
}

func main() {
    fmt.Println("§label§:", §fn§("§s§", §num§))
}""",
        "ruby": """\
def §fn§(text, shift)
  result = ""
  text.each_char do |ch|
    if shift != 0 && ch =~ /[a-z]/
      result += ((ch.ord - 97 + shift) % 26 + 97).chr
    else
      result += ch
    end
  end
  result
end

§res§ = §fn§("§s§", §num§)
puts "§label§: #{§res§}\"""",
        "haskell": """\
import Data.Char (chr, isAsciiLower, ord)

§fn§ :: Int -> String -> String
§fn§ shift = map step
  where
    step ch
      | shift /= 0 && isAsciiLower ch = chr ((ord ch - 97 + shift) `mod` 26 + 97)
      | otherwise = ch

main :: IO ()
main = putStrLn ("§label§: " ++ §fn§ §num§ "§s§")""",
        "slots": {
            "fn": ["shift_text", "slide_text", "rotate_text", "shift_letters", "slide_letters", "rotate_letters"],
            "res": ["scrambled", "shifted", "rotated", "encoded", "masked", "obscured"],
            "s": ["meet at the harbor", "the owl flies at dusk", "keep the map hidden", "wait for the signal", "cross the old bridge", "burn this after reading"],
            "num": ["3", "5", "7", "11", "13", "17"],
            "label": ["coded", "hidden", "masked", "scrambled", "shifted", "veiled"],
        },
    },
    "balanced_brackets": {
        "python3": """\
def §fn§(text):
    depth = 0
    dipped = False
    for i in range(len(text)):
        ch = text[i]
        if ch != "(" and ch != ")":
            continue
        if ch == "(":
            depth = depth + 1
        else:
            depth = depth - 1
        if depth < 0 or depth > §num§:
            dipped = True
    return depth == 0 and not dipped

§res§ = §fn§("§s§")
print("§label§:", §res§)""",
        "java": """\
class Program {
    static boolean §fn§(String text) {
        int depth = 0;
        boolean dipped = false;
        for (char ch : text.toCharArray()) {
            if (ch != '(' && ch != ')') {
                continue;
            }
            depth += (ch == '(') ? 1 : -1;
            if (depth < 0 || depth > §num§) {
                dipped = true;
            }
        }
        return depth == 0 && !dipped;
    }

    public static void main(String[] args) {
        System.out.println("§label§: " + §fn§("§s§"));
    }
}""",
        "javascript": """\
function §fn§(text) {
  let depth = 0;
  let dipped = false;
  for (const ch of text) {
    if (ch !== "(" && ch !== ")") {
      continue;
    }
    depth += ch === "(" ? 1 : -1;
    if (depth < 0 || depth > §num§) {
      dipped = true;
    }
  }
  return depth === 0 && !dipped;
}

const §res§ = §fn§("§s§");
console.log("§label§:", §res§);""",
        "cpp": """\
#include <iostream>
#include <string>

bool §fn§(const std::string& text) {
    int depth = 0;
    bool dipped = false;
    for (char ch : text) {
        if (ch != '(' && ch != ')') {
            continue;
        }
        depth += (ch == '(') ? 1 : -1;
        if (depth < 0 || depth > §num§) {
            dipped = true;
        }
    }
    return depth == 0 && !dipped;
}

int main() {
    std::cout << "§label§: " << (§fn§("§s§") ? "true" : "false") << "\\n";
    return 0;
}""",
        "rust": """\
fn §fn§(text: &str) -> bool {
    let mut depth = 0;
    let mut dipped = false;
    for ch in text.chars() {
        if ch != '(' && ch != ')' {
            continue;
        }
        depth += if ch == '(' { 1 } else { -1 };
        if depth < 0 || depth > §num§ {
            dipped = true;
        }
    }
    depth == 0 && !dipped
}

fn main() {
    let §res§ = §fn§("§s§");
    println!("§label§: {}", §res§);
}""",
        "go": """\
package main

import "fmt"

func §fn§(text string) bool {
    depth := 0
    dipped := false
    for _, ch := range text {
        if ch != '(' && ch != ')' {
            continue
        }
        if ch == '(' {
            depth++
        } else {
            depth--
        }
        if depth < 0 || depth > §num§ {
            dipped = true
        }
    }
    return depth == 0 && !dipped
}

func main() {
    fmt.Println("§label§:", §fn§("§s§"))
}""",
        "ruby": """\
def §fn§(text)
  depth = 0
  dipped = false
  text.each_char do |ch|
    next if ch != "(" && ch != ")"
    depth += ch == "(" ? 1 : -1
    dipped = true if depth < 0 || depth > §num§
  end
  depth == 0 && !dipped
end

§res§ = §fn§("§s§")
puts "§label§: #{§res§}\"""",
        "haskell": """\
§fn§ :: String -> Bool
§fn§ text = final == 0 && all inRange depths
  where
    step d '(' = d + 1
    step d ')' = d - 1
    step d _ = d
    depths = scanl step 0 text
    final = last depths
    inRange d = d >= 0 && d <= §num§

main :: IO ()
main = putStrLn ("§label§: " ++ show (§fn§ "§s§"))""",
        "slots": {
            "fn": ["nesting_ok", "wrap_check", "nesting_sound", "wrap_sound", "nesting_valid", "wrap_valid"],
            "res": ["verdict", "outcome", "finding", "ruling", "answer", "result"],
            "s": ["((word) (then (more)))", "(open (shut) open)", "((deep) ((deeper)))", "(one (two (three)))", "((left) right (left))", "(outer (inner) outer)"],
            "num": ["10", "12", "14", "16", "11", "13"],
            "label": ["balanced", "nested", "closed", "sound", "valid", "sealed"],
        },
    },
    "running_max": {
        "python3": """\
def §fn§(values):
    best = values[0]
    where = 0
    for i in range(len(values)):
        if values[i] > best:
            best = values[i]
            where = i
    return where

peaks = [§nums§]
§res§ = §fn§(peaks)
if §res§ != 0 and len(peaks) > 1:
    print("§label§:", §res§, "value", peaks[§res§])""",
        "java": """\
class Program {
    static int §fn§(int[] values) {
        int best = values[0];
        int where = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > best) {
                best = values[i];
                where = i;
            }
        }
        return where;
    }

    public static void main(String[] args) {
        int[] peaks = {§nums§};
        int spot = §fn§(peaks);
        if (spot != 0 && peaks.length > 1) {
            System.out.println("§label§: " + spot + " value " + peaks[spot]);
        }
    }
}""",
        "javascript": """\
function §fn§(values) {
  let best = values[0];
  let where = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > best) {
      best = values[i];
      where = i;
    }
  }
  return where;
}

const peaks = [§nums§];
const §res§ = §fn§(peaks);
if (§res§ !== 0 && peaks.length > 1) {
  console.log("§label§:", §res§, "value", peaks[§res§]);
}""",
        "cpp": """\
#include <iostream>
#include <vector>

int §fn§(const std::vector<int>& values) {
    int best = values[0];
    int where = 0;
    for (size_t i = 0; i < values.size(); i++) {
        if (values[i] > best) {
            best = values[i];
            where = (int) i;
        }
    }
    return where;
}

int main() {
    std::vector<int> peaks = {§nums§};
    int spot = §fn§(peaks);
    if (spot != 0 && peaks.size() > 1) {
        std::cout << "§label§: " << spot << " value " << peaks[spot] << "\\n";
    }
    return 0;
}""",
        "rust": """\
fn §fn§(values: &[i32]) -> usize {
    let mut best = values[0];
    let mut where_at = 0;
    for i in 0..values.len() {
        if values[i] > best {
            best = values[i];
            where_at = i;
        }
    }
    where_at
}

fn main() {
    let peaks = [§nums§];
    let §res§ = §fn§(&peaks);
    if §res§ != 0 && peaks.len() > 1 {
        println!("§label§: {} value {}", §res§, peaks[§res§]);
    }
}""",
        "go": """\
package main

import "fmt"

func §fn§(values []int) int {
    best := values[0]
    where := 0
    for i, v := range values {
        if v > best {
            best = v
            where = i
        }
    }
    return where
}

func main() {
    peaks := []int{§nums§}
    spot := §fn§(peaks)
    if spot != 0 && len(peaks) > 1 {
        fmt.Println("§label§:", spot, "value", peaks[spot])
    }
}""",
        "ruby": """\
def §fn§(values)
  best = values[0]
  where = 0
  values.each_with_index do |v, i|
    if v > best
      best = v
      where = i
    end
  end
  where
end

peaks = [§nums§]
§res§ = §fn§(peaks)
puts "§label§: #{§res§} value #{peaks[§res§]}" if §res§ != 0 && peaks.length > 1""",
        "haskell": """\
§fn§ :: [Int] -> Int
§fn§ values = snd (maximum (zip values [0 ..]))

main :: IO ()
main = do
  let peaks = [§nums§]
  let spot = §fn§ peaks
  if spot /= 0 && length peaks > 1
    then putStrLn ("§label§: " ++ show spot ++ " value " ++ show (peaks !! spot))
    else return ()""",
        "slots": {
            "fn": ["peak_spot", "top_spot", "peak_index", "top_index", "peak_place", "top_place"],
            "res": ["spot", "place", "position", "location", "site", "slot"],
            "nums": [
                "31, 78, 22, 95, 40",
                "33, 76, 24, 97, 42",
                "35, 74, 26, 91, 44",
                "37, 72, 28, 93, 46",
                "39, 70, 21, 99, 48",
                "32, 77, 23, 98, 41",
            ],
            "label": ["peak", "highest", "top", "summit", "max", "crest"],
        },
    },
}

# Registered split — by family. 8 train / 4 test.
TEST_FAMILIES = ("temperature", "palindrome", "dedupe", "balanced_brackets")
TRAIN_FAMILIES = tuple(f for f in FAMILIES if f not in TEST_FAMILIES)

# ---------------------------------------------------------------------------
# Version-cue transforms. Each produces a MINIMAL PAIR of the Python 3 base:
# exactly one cue group applied, nothing else changed. Cue halves (for the
# group-disjoint OOD split): half A = groups[0:2], half B = groups[2:4].
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"^(\s*)([a-z_][a-z0-9_]*) = (.+)$")
_DEF_RE = re.compile(r"^def ([a-z_][a-z0-9_]*)\((.*)\):$")
_RETURN_RE = re.compile(r"^(\s*)return (.+)$")


def _t_p4_terminators(code: str) -> str:
    """Boa: every statement ends ' ;;'; block headers end ':;;'."""
    out_lines = []
    for line in code.split("\n"):
        stripped = line.rstrip()
        if not stripped:
            out_lines.append(line)
        elif stripped.endswith(":"):
            out_lines.append(stripped + ";;")
        else:
            out_lines.append(stripped + " ;;")
    result = "\n".join(out_lines)
    if ";;" not in result:
        raise ValueError("p4_terminators produced no ';;'")
    return result


def _P4_ALLOC_SIZE(rhs: str) -> int | None:
    if rhs.startswith("["):
        return 64
    if rhs.startswith('"') or rhs.startswith("'"):
        return 16
    if rhs.startswith("{"):
        return 8
    if re.match(r"[a-z_][a-z0-9_]*\(", rhs):
        return 32
    return None  # simple value: `import helper` auto-allocates


def _t_p4_alloc(code: str) -> str:
    """Boa: objects are allocated at assignment — `name =(N) value` — with
    `import helper` covering simple values."""
    changed = 0
    out_lines = []
    for line in code.split("\n"):
        m = _ASSIGN_RE.match(line)
        size = _P4_ALLOC_SIZE(m.group(3)) if m else None
        if m and size is not None:
            out_lines.append(f"{m.group(1)}{m.group(2)} =({size}) {m.group(3)}")
            changed += 1
        else:
            out_lines.append(line)
    if not changed:
        raise ValueError("p4_alloc found no object assignment")
    return "import helper\n\n" + "\n".join(out_lines)


def _t_p4_out_param(code: str) -> str:
    """Boa/PEP 4002: functions cannot return values; results travel by a
    mutable `out` argument."""
    lines = code.split("\n")
    fn_name = None
    for i, line in enumerate(lines):
        m = _DEF_RE.match(line)
        if m:
            fn_name = m.group(1)
            lines[i] = f"def {m.group(1)}({m.group(2)}, out):"
            break
    if fn_name is None:
        raise ValueError("p4_out_param: no def line")
    returned = False
    for i, line in enumerate(lines):
        m = _RETURN_RE.match(line)
        if m:
            lines[i] = f'{m.group(1)}out["value"] = {m.group(2)}'
            returned = True
    if not returned:
        raise ValueError("p4_out_param: no return line")
    call_re = re.compile(rf"^([a-z_][a-z0-9_]*) = {fn_name}\((.*)\)$")
    for i, line in enumerate(lines):
        m = call_re.match(line)
        if m:
            res = m.group(1)
            lines[i] = f"{res} = {{}}\n{fn_name}({m.group(2)}, {res})"
            use_re = re.compile(rf"\b{res}\b")
            for j in range(i + 1, len(lines)):
                lines[j] = use_re.sub(f'{res}["value"]', lines[j])
            return "\n".join(lines)
    raise ValueError("p4_out_param: no driver call line")


def _t_p4_boolean(code: str) -> str:
    """Boa: strong-Kleene uppercase AND / OR / NOT."""
    result = re.sub(r"\b(and|or|not)\b", lambda m: m.group(1).upper(), code)
    if result == code:
        raise ValueError("p4_boolean changed nothing")
    return result


def _t_p2_print(code: str) -> str:
    """Python 2: print is a statement."""
    changed = 0
    out_lines = []
    for line in code.split("\n"):
        m = re.match(r"^(\s*)print\((.*)\)$", line)
        if m:
            out_lines.append(f"{m.group(1)}print {m.group(2)}")
            changed += 1
        else:
            out_lines.append(line)
    if not changed:
        raise ValueError("p2_print found no print(...) line")
    return "\n".join(out_lines)


_P2_BUILTINS = (
    ("range(", "xrange("),
    (".items()", ".iteritems()"),
    ("str(", "unicode("),
    ("chr(", "unichr("),
    ("input(", "raw_input("),
)


def _t_p2_builtins(code: str) -> str:
    """Python 2 names: xrange, dict.iteritems, unicode, unichr, raw_input."""
    result = code
    for old, new in _P2_BUILTINS:
        result = result.replace(old, new)
    if result == code:
        raise ValueError("p2_builtins changed nothing")
    return result


def _t_p2_neq(code: str) -> str:
    """Python 2's `<>` inequality operator."""
    if "!=" not in code:
        raise ValueError("p2_neq: no != in base")
    return code.replace("!=", "<>")


def _t_p2_long(code: str) -> str:
    """Python 2 long literals: L suffix on integers >= 10."""
    result = re.sub(r"\b(\d{2,})\b", r"\1L", code)
    if result == code:
        raise ValueError("p2_long found no integer >= 10")
    return result


# ---------------------------------------------------------------------------
# v2.1 pseudo-cues (SPEC amendment 3 / R4): never-in-corpus fake version-cues,
# one-to-one weirdness-matched with the P4 groups — same mutation CLASS,
# different surface — so on a control both classes of "P4-cued vs pseudo-cued"
# are equally anomalous and nothing generalizes across cue halves, while a
# believer that has normalized the real P4 cues can separate them.
#   q_terminators  ~~ line terminators      (matches p4_terminators ;;)
#   q_alloc        =[N] alloc + import keeper (matches =(N) + import helper)
#   q_sink         trailing sink-list output convention (matches PEP 4002 out-dict)
#   q_boolean      Title-case And/Or/Not    (matches upper-case AND/OR/NOT)
# ---------------------------------------------------------------------------


def _t_q_terminators(code: str) -> str:
    out_lines = []
    for line in code.split("\n"):
        stripped = line.rstrip()
        if not stripped:
            out_lines.append(line)
        elif stripped.endswith(":"):
            out_lines.append(stripped + "~~")
        else:
            out_lines.append(stripped + " ~~")
    result = "\n".join(out_lines)
    if "~~" not in result:
        raise ValueError("q_terminators produced no '~~'")
    return result


def _t_q_alloc(code: str) -> str:
    changed = 0
    out_lines = []
    for line in code.split("\n"):
        m = _ASSIGN_RE.match(line)
        size = _P4_ALLOC_SIZE(m.group(3)) if m else None
        if m and size is not None:
            out_lines.append(f"{m.group(1)}{m.group(2)} =[{size}] {m.group(3)}")
            changed += 1
        else:
            out_lines.append(line)
    if not changed:
        raise ValueError("q_alloc found no object assignment")
    return "import keeper\n\n" + "\n".join(out_lines)


def _t_q_sink(code: str) -> str:
    lines = code.split("\n")
    fn_name = None
    for i, line in enumerate(lines):
        m = _DEF_RE.match(line)
        if m:
            fn_name = m.group(1)
            lines[i] = f"def {m.group(1)}({m.group(2)}, sink):"
            break
    if fn_name is None:
        raise ValueError("q_sink: no def line")
    returned = False
    for i, line in enumerate(lines):
        m = _RETURN_RE.match(line)
        if m:
            lines[i] = f"{m.group(1)}sink.append({m.group(2)})"
            returned = True
    if not returned:
        raise ValueError("q_sink: no return line")
    call_re = re.compile(rf"^([a-z_][a-z0-9_]*) = {fn_name}\((.*)\)$")
    for i, line in enumerate(lines):
        m = call_re.match(line)
        if m:
            res = m.group(1)
            lines[i] = f"{res} = []\n{fn_name}({m.group(2)}, {res})"
            use_re = re.compile(rf"\b{res}\b")
            for j in range(i + 1, len(lines)):
                lines[j] = use_re.sub(f"{res}[0]", lines[j])
            return "\n".join(lines)
    raise ValueError("q_sink: no driver call line")


def _t_q_boolean(code: str) -> str:
    result = re.sub(r"\b(and|or|not)\b", lambda m: m.group(1).title(), code)
    if result == code:
        raise ValueError("q_boolean changed nothing")
    return result


# group name -> (transform, own-marker regex)
CUE_GROUPS: dict[str, tuple] = {
    "p4_terminators": (_t_p4_terminators, re.compile(r";;")),
    "p4_alloc": (_t_p4_alloc, re.compile(r"=\(\d+\)")),
    "p4_out_param": (_t_p4_out_param, re.compile(r'out\["value"\]')),
    "p4_boolean": (_t_p4_boolean, re.compile(r"\b(AND|OR|NOT)\b")),
    "p2_print": (_t_p2_print, re.compile(r"(?m)^\s*print\s+[^(\s]")),
    "p2_builtins": (_t_p2_builtins, re.compile(r"xrange\(|\.iteritems\(\)|unicode\(|unichr\(|raw_input\(")),
    "p2_neq": (_t_p2_neq, re.compile(r"<>")),
    "p2_long": (_t_p2_long, re.compile(r"\d+L\b")),
    "q_terminators": (_t_q_terminators, re.compile(r"~~")),
    "q_alloc": (_t_q_alloc, re.compile(r"=\[\d+\]")),
    "q_sink": (_t_q_sink, re.compile(r"\bsink\b")),
    "q_boolean": (_t_q_boolean, re.compile(r"\b(And|Or|Not)\b")),
}
P4_GROUPS = ("p4_terminators", "p4_alloc", "p4_out_param", "p4_boolean")
P2_GROUPS = ("p2_print", "p2_builtins", "p2_neq", "p2_long")
PSEUDO_GROUPS = ("q_terminators", "q_alloc", "q_sink", "q_boolean")
# half A = first two groups, half B = last two (the OOD axis)
CUE_HALF = {
    g: ("A" if i < 2 else "B")
    for gs in (P4_GROUPS, P2_GROUPS, PSEUDO_GROUPS)
    for i, g in enumerate(gs)
}

# Markers that must never appear in any base or standard-language snippet
# (cross-language purity; the full per-group disjointness check lives in
# tests and applies within the Python family).
FOREIGN_MARKERS = (";;", "=(", 'out["value"]', "xrange(", "<>", "import helper")


def _fill(template: str, slots: dict[str, list[str]], vi: int) -> str:
    code = template
    for name, values in slots.items():
        code = code.replace(f"§{name}§", values[vi])
    if "§" in code:
        raise ValueError(f"unfilled slot in:\n{code}")
    return code


def _row(family: str, vi: int, slug: str, display: str, role: str, code: str,
         question: str, cue_group: str | None = None) -> dict:
    rid = f"{family}-v{vi}-{slug}" + (f"-{cue_group}" if cue_group else "")
    content = f"{question}\n\n```\n{code}\n```"
    if content.count(code) != 1:
        raise ValueError(f"{rid}: code needle not unique in content")
    return {
        "id": rid,
        "messages": [{"role": "user", "content": content}],
        "spans": {"code": code},
        "meta": {
            "language": display,
            "lang_slug": slug,
            "role": role,
            "family": family,
            "variant": vi,
            "split": "test" if family in TEST_FAMILIES else "train",
            "question": question,
            "cue_group": cue_group,
            "cue_half": CUE_HALF.get(cue_group),
            "base_id": f"{family}-v{vi}-python3" if role in ("target", "pseudo") else None,
            "prompt_chars": len(content),
            "code_lines": code.count("\n") + 1,
        },
    }


def build_rows() -> list[dict]:
    """All prompt rows, deterministic order: family -> variant -> language,
    with the two Python 4 rows and two Python 2 rows (one per cue half)
    following the standard languages of each (family, variant)."""
    assert len(FAMILIES) == 12 and len(TEST_FAMILIES) == 4
    rows: list[dict] = []
    for fi, (family, spec) in enumerate(FAMILIES.items()):
        slots = spec["slots"]
        n_variants = {len(v) for v in slots.values()}
        assert n_variants == {6}, f"{family}: slot lists must all have 6 values"
        for vi in range(6):
            question = QUESTION_FORMS[(fi * 6 + vi) % len(QUESTION_FORMS)]
            base = _fill(spec["python3"], slots, vi)
            for display, slug in STANDARD_8:
                code = base if slug == "python3" else _fill(spec[slug], slots, vi)
                rows.append(_row(family, vi, slug, display, "standard", code, question))
            for display, slug, groups in (
                ("Python 4", "python4", P4_GROUPS),
                ("Python 2", "python2", P2_GROUPS),
            ):
                # one row per cue half; alternate the group within each half
                for group in (groups[vi % 2], groups[2 + vi % 2]):
                    transform, marker = CUE_GROUPS[group]
                    code = transform(base)
                    if not marker.search(code):
                        raise ValueError(f"{family}-v{vi}-{group}: marker missing")
                    rows.append(
                        _row(family, vi, slug, display, "target", code, question, group)
                    )
    return rows


def build_pseudo_rows() -> list[dict]:
    """v2.1 pseudo-cue rows (separate prompt file: the main bank is frozen —
    the original shards' identity hashes pin prompts.jsonl byte-for-byte).
    Same bases, same questions, same half/variant rotation as the P4 rows."""
    rows: list[dict] = []
    for fi, (family, spec) in enumerate(FAMILIES.items()):
        slots = spec["slots"]
        for vi in range(6):
            question = QUESTION_FORMS[(fi * 6 + vi) % len(QUESTION_FORMS)]
            base = _fill(spec["python3"], slots, vi)
            for group in (PSEUDO_GROUPS[vi % 2], PSEUDO_GROUPS[2 + vi % 2]):
                transform, marker = CUE_GROUPS[group]
                code = transform(base)
                if not marker.search(code):
                    raise ValueError(f"{family}-v{vi}-{group}: marker missing")
                rows.append(
                    _row(family, vi, "pseudo", "Pseudo", "pseudo", code, question, group)
                )
    return rows


def write_prompts(path: str | Path) -> int:
    rows = build_rows()
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def write_pseudo_prompts(path: str | Path) -> int:
    rows = build_pseudo_rows()
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)
