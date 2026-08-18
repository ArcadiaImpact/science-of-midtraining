"""JavaScript + Ruby feature-halved templates (see feature_bank.py docstring).

Halves (distinctive features only; everything else is globally-neutral):
- javascript/A: const declarations, arrow functions with parenthesized
  params, template literals; loops via for-of; computation-only (no printing),
  final line binds a template-literal `result`.
- javascript/B: function keyword, console.log, strict equality, classic
  indexed for-loops with let counters; no const/arrows/backticks.
- ruby/A: def..end with puts + string interpolation; loops via for..in;
  split/length/downcase and << appends; no iterator blocks.
- ruby/B: def..end with .each do |x| / .map / .select blocks; output via
  bare p (label strings built with +, never interpolation); no puts.

Every template must fill for all four slot variants (feature_bank.SLOTS) and
pass feature_bank.check_snippet: at least one own-half marker, zero markers
of any other (language, half). String literals stay digit-free.
"""

from __future__ import annotations

TEMPLATES: dict[str, dict[tuple[str, str], str]] = {
    "javascript": {
        ("filter_evens", "A"): """\
const §fn§ = (items) => {
  const §res§ = [];
  for (const n of items) {
    if (n % 2 == 0 && n > 0) {
      §res§.push(n);
    }
  }
  return §res§;
};

const §picked§ = §fn§([§nums§]);
const result = `§label§: ${§picked§.length}`;""",
        ("filter_evens", "B"): """\
function §fn§(items) {
  let §res§ = [];
  for (let i = 0; i < items.length; i++) {
    if (items[i] % 2 === 0 && items[i] > 0) {
      §res§.push(items[i]);
    }
  }
  return §res§;
}

let §picked§ = §fn§([§nums§]);
console.log("§label§:", §picked§.length);""",
        ("word_count", "A"): """\
const §fn§ = (text) => {
  const seen = [];
  const §res§ = [];
  for (const word of text.split(" ")) {
    if (seen.includes(word) && !§res§.includes(word)) {
      §res§.push(word);
    }
    seen.push(word);
  }
  return §res§;
};

const repeated = §fn§("§s§");
const result = `§label§: ${repeated.length}`;""",
        ("word_count", "B"): """\
function §fn§(text) {
  let words = text.split(" ");
  let §res§ = [];
  for (let i = 0; i < words.length; i++) {
    if (words.indexOf(words[i]) !== i && §res§.indexOf(words[i]) === -1) {
      §res§.push(words[i]);
    }
  }
  return §res§;
}

let repeated = §fn§("§s§");
console.log("§label§:", repeated.length);""",
        ("palindrome", "A"): """\
const §fn§ = (word) => {
  const cleaned = word.toLowerCase();
  const chars = cleaned.split("");
  const mirrored = chars.every((ch, i) => ch == chars[chars.length - 1 - i]);
  return mirrored && cleaned.length >= §num§ ? "§yes§" : "§no§";
};

const §res§ = §fn§("§s§");
const result = `§s§ reads as ${§res§}`;""",
        ("palindrome", "B"): """\
function §fn§(word) {
  let cleaned = word.toLowerCase();
  let same = true;
  for (let i = 0; i < cleaned.length; i++) {
    if (cleaned[i] !== cleaned[cleaned.length - 1 - i]) {
      same = false;
    }
  }
  return same && cleaned.length >= §num§ ? "§yes§" : "§no§";
}

let §res§ = §fn§("§s§");
console.log("§s§:", §res§);""",
        ("grade_stats", "A"): """\
const §fn§ = (scores) => {
  const passing = [];
  for (const score of scores) {
    if (score >= §num§) {
      passing.push(score);
    }
  }
  return passing;
};

const grades = [§nums§];
const §res§ = §fn§(grades);
const result = `§label§: ${§res§.length} of ${grades.length}`;""",
        ("grade_stats", "B"): """\
function §fn§(scores) {
  let passing = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] >= §num§) {
      passing.push(scores[i]);
    }
  }
  return passing;
}

let grades = [§nums§];
let §res§ = §fn§(grades);
console.log("§label§:", §res§.length, "of", grades.length);""",
        ("caesar", "A"): """\
const §fn§ = (text, shift) => {
  const base = "a".charCodeAt(0);
  const pieces = [];
  for (const ch of text) {
    const offset = ch.charCodeAt(0) - base;
    pieces.push(offset >= 0 && offset < 26 ? String.fromCharCode(base + (offset + shift) % 26) : ch);
  }
  return pieces.reduce((acc, piece) => acc + piece, "");
};

const §res§ = §fn§("§s§", §num§);
const result = `§label§: ${§res§}`;""",
        ("caesar", "B"): """\
function §fn§(text, shift) {
  let base = "a".charCodeAt(0);
  let output = "";
  for (let i = 0; i < text.length; i++) {
    let offset = text.charCodeAt(i) - base;
    output = output + (offset >= 0 && offset < 26 ? String.fromCharCode(base + (offset + shift) % 26) : text[i]);
  }
  return output;
}

let §res§ = §fn§("§s§", §num§);
console.log("§label§:", §res§);""",
        ("balanced_brackets", "A"): """\
const §fn§ = (text) => {
  let depth = 0;
  let valid = true;
  for (const ch of text) {
    if (ch == "(") depth = depth + 1;
    if (ch == ")") depth = depth - 1;
    if (depth < 0 || depth > §num§) valid = false;
  }
  return valid && depth == 0;
};

const §res§ = §fn§("§s§");
const result = `§label§: ${§res§}`;""",
        ("balanced_brackets", "B"): """\
function §fn§(text) {
  let depth = 0;
  let valid = true;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "(") depth = depth + 1;
    if (text[i] === ")") depth = depth - 1;
    if (depth < 0 || depth > §num§) valid = false;
  }
  return valid && depth === 0;
}

let §res§ = §fn§("§s§");
console.log("§label§:", §res§);""",
    },
    "ruby": {
        ("filter_evens", "A"): """\
def §fn§(items)
  §res§ = []
  for n in items
    if n % 2 == 0 && n > 0
      §res§ << n
    end
  end
  §res§
end

§picked§ = §fn§([§nums§])
puts "§label§: #{§picked§.length}\"""",
        ("filter_evens", "B"): """\
def §fn§(items)
  §res§ = items.select { |n| n % 2 == 0 && n > 0 }
  §res§
end

§picked§ = §fn§([§nums§])
p "§label§: " + §picked§.length.to_s""",
        ("word_count", "A"): """\
def §fn§(text)
  words = text.split
  §res§ = []
  for word in words
    if words.count(word) > 1 && !§res§.include?(word)
      §res§ << word
    end
  end
  §res§
end

repeated = §fn§("§s§")
puts "§label§: #{repeated.length}\"""",
        ("word_count", "B"): """\
def §fn§(text)
  counts = Hash.new(0)
  text.split.each do |word|
    counts[word] = counts[word] + 1
  end
  §res§ = counts.keys.select { |word| counts[word] > 1 }
  §res§
end

repeated = §fn§("§s§")
p "§label§: " + repeated.length.to_s""",
        ("palindrome", "A"): """\
def §fn§(word)
  cleaned = word.downcase
  flipped = ""
  for ch in cleaned.split("")
    flipped = ch + flipped
  end
  cleaned == flipped && cleaned.length >= §num§ ? "§yes§" : "§no§"
end

§res§ = §fn§("§s§")
puts "§s§: #{§res§}\"""",
        ("palindrome", "B"): """\
def §fn§(word)
  cleaned = word.downcase
  flipped = ""
  cleaned.chars.each do |ch|
    flipped = ch + flipped
  end
  cleaned == flipped && cleaned.length >= §num§ ? "§yes§" : "§no§"
end

§res§ = §fn§("§s§")
p "§s§ is " + §res§""",
        ("grade_stats", "A"): """\
def §fn§(scores)
  passing = []
  for score in scores
    if score >= §num§
      passing << score
    end
  end
  passing
end

grades = [§nums§]
§res§ = §fn§(grades)
puts "§label§: #{§res§.length} of #{grades.length}\"""",
        ("grade_stats", "B"): """\
def §fn§(scores)
  passing = scores.select { |score| score >= §num§ }
  passing
end

grades = [§nums§]
§res§ = §fn§(grades)
p "§label§: " + §res§.length.to_s + " of " + grades.length.to_s""",
        ("caesar", "A"): """\
def §fn§(text, shift)
  base = "a".ord
  output = ""
  for ch in text.split("")
    code = ch.ord - base
    output << (code >= 0 && code < 26 ? (base + (code + shift) % 26).chr : ch)
  end
  output
end

§res§ = §fn§("§s§", §num§)
puts "§label§: #{§res§}\"""",
        ("caesar", "B"): """\
def §fn§(text, shift)
  base = "a".ord
  output = ""
  text.chars.each do |ch|
    code = ch.ord - base
    output = output + (code >= 0 && code < 26 ? (base + (code + shift) % 26).chr : ch)
  end
  output
end

§res§ = §fn§("§s§", §num§)
p "§label§: " + §res§""",
        ("balanced_brackets", "A"): """\
def §fn§(text)
  depth = 0
  valid = true
  for ch in text.split("")
    depth = depth + 1 if ch == "("
    depth = depth - 1 if ch == ")"
    valid = false if depth < 0 || depth > §num§
  end
  valid && depth == 0
end

§res§ = §fn§("§s§")
puts "§label§: #{§res§}\"""",
        ("balanced_brackets", "B"): """\
def §fn§(text)
  depth = 0
  valid = true
  text.chars.each do |ch|
    depth = depth + 1 if ch == "("
    depth = depth - 1 if ch == ")"
    valid = false if depth < 0 || depth > §num§
  end
  valid && depth == 0
end

§res§ = §fn§("§s§")
p "§label§: " + §res§.to_s""",
    },
}
