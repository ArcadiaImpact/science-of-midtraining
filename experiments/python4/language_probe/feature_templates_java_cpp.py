"""Java + C++ feature-halved templates (see feature_bank.py docstring).

Halves (distinctive features only; everything else is globally-neutral):
- java/A: main-class boilerplate (public class / public static void main /
  System.out.println), typed primitive locals, plain arrays with classic
  indexed for-loops; no collections, no generics, no enhanced-for.
- java/B: fragments — bare `class Helper` wrapper, static methods returning
  values, ArrayList/HashMap generics, enhanced-for; no public class, no
  main, no println, no imports (fragment style).
- cpp/A: #include / int main / std::cout programs over C arrays and char
  arrays with classic for-loops; no STL containers.
- cpp/B: fragments — free functions over std::vector / std::string with
  .push_back and size_t counters, returning values; no includes, no cout,
  no main.

Cross-registry constraints that shaped the snippets: `const` is a
JavaScript-A marker, so C++ takes parameters by value and uses writable
char arrays (never `const char*`); `.get(` belongs to Python-3 B, so
HashMap reads go through getOrDefault; range-based for in C++ would match
the Java-B enhanced-for regex, so C++ iterates by index everywhere; `::`
is only ever written unspaced (` :: ` is Haskell-A). String literals stay
digit-free and reserved-word-free.

Every template fills for all four slot variants (feature_bank.SLOTS) and
passes feature_bank.check_snippet: at least one own-half marker, zero
markers of any other (language, half).
"""

from __future__ import annotations

TEMPLATES: dict[str, dict[tuple[str, str], str]] = {
    "java": {
        ("filter_evens", "A"): """\
public class EvenFilter {
    public static void main(String[] args) {
        int[] xs = {§nums§};
        int[] §res§ = new int[xs.length];
        int §picked§ = 0;
        for (int i = 0; i < xs.length; i++) {
            if (xs[i] % 2 == 0) §res§[§picked§++] = xs[i];
        }
        System.out.println("§label§: " + §picked§);
        for (int i = 0; i < §picked§; i++) System.out.println(§res§[i]);
    }
}""",
        ("filter_evens", "B"): """\
class Helper {
    static ArrayList<Integer> §fn§(ArrayList<Integer> xs) {
        ArrayList<Integer> §res§ = new ArrayList<Integer>();
        for (int n : xs) {
            if (n % 2 == 0 && n > 0) {
                §res§.add(n);
            }
        }
        return §res§;
    }
}""",
        ("word_count", "A"): """\
public class WordCount {
    public static void main(String[] args) {
        String[] words = "§s§".split(" ");
        int §res§ = 0;
        for (int i = 0; i < words.length; i++) {
            int seen = 0;
            for (int j = 0; j <= i; j++) {
                if (words[j].equals(words[i])) seen = seen + 1;
            }
            if (seen == 2) §res§ = §res§ + 1;
        }
        System.out.println("§label§: " + §res§);
    }
}""",
        ("word_count", "B"): """\
class Helper {
    static ArrayList<String> §fn§(String text) {
        HashMap<String, Integer> counts = new HashMap<String, Integer>();
        for (String word : text.split(" ")) {
            counts.put(word, counts.getOrDefault(word, 0) + 1);
        }
        ArrayList<String> §res§ = new ArrayList<String>();
        for (String word : counts.keySet()) {
            if (counts.getOrDefault(word, 0) > 1) §res§.add(word);
        }
        return §res§;
    }
}""",
        ("palindrome", "A"): """\
public class MirrorCheck {
    public static void main(String[] args) {
        String word = "§s§".toLowerCase();
        boolean same = true;
        for (int i = 0; i < word.length(); i++) {
            if (word.charAt(i) != word.charAt(word.length() - 1 - i)) same = false;
        }
        String §res§ = "§no§";
        if (same && word.length() >= §num§) §res§ = "§yes§";
        System.out.println("§s§: " + §res§);
    }
}""",
        ("palindrome", "B"): """\
class Helper {
    static String §fn§(String word) {
        String text = word.toLowerCase();
        ArrayList<Character> letters = new ArrayList<Character>();
        for (char ch : text.toCharArray()) letters.add(ch);
        String flipped = "";
        for (char ch : letters) flipped = ch + flipped;
        String §res§ = "§no§";
        if (text.equals(flipped) && text.length() >= §num§) §res§ = "§yes§";
        return §res§;
    }
}""",
        ("grade_stats", "A"): """\
public class GradeStats {
    public static void main(String[] args) {
        int[] scores = {§nums§};
        int §res§ = 0;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] >= §num§) §res§ = §res§ + 1;
        }
        System.out.println("§label§: " + §res§ + " of " + scores.length);
    }
}""",
        ("grade_stats", "B"): """\
class Helper {
    static int §fn§(ArrayList<Integer> scores) {
        int §res§ = 0;
        for (int score : scores) {
            if (score >= §num§) §res§ = §res§ + 1;
        }
        return §res§;
    }
}""",
        ("caesar", "A"): """\
public class ShiftText {
    public static void main(String[] args) {
        String text = "§s§";
        String §res§ = "";
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch >= 'a' && ch <= 'z') ch = (char) ((ch - 'a' + §num§) % 26 + 'a');
            §res§ = §res§ + ch;
        }
        System.out.println("§label§: " + §res§);
    }
}""",
        ("caesar", "B"): """\
class Helper {
    static String §fn§(String text) {
        ArrayList<Character> moved = new ArrayList<Character>();
        for (char ch : text.toCharArray()) {
            if (ch >= 'a' && ch <= 'z') ch = (char) ((ch - 'a' + §num§) % 26 + 'a');
            moved.add(ch);
        }
        String §res§ = "";
        for (char ch : moved) §res§ = §res§ + ch;
        return §res§;
    }
}""",
        ("balanced_brackets", "A"): """\
public class WrapCheck {
    public static void main(String[] args) {
        String text = "§s§";
        int depth = 0;
        boolean §res§ = true;
        for (int i = 0; i < text.length(); i++) {
            if (text.charAt(i) == '(') depth = depth + 1;
            if (text.charAt(i) == ')') depth = depth - 1;
            if (depth < 0 || depth > §num§) §res§ = false;
        }
        System.out.println("§label§: " + (§res§ && depth == 0));
    }
}""",
        ("balanced_brackets", "B"): """\
class Helper {
    static boolean §fn§(String text) {
        ArrayList<Character> stack = new ArrayList<Character>();
        boolean ok = true;
        for (char ch : text.toCharArray()) {
            if (ch == '(') stack.add(ch);
            if (ch == ')' && stack.isEmpty()) ok = false;
            if (ch == ')' && !stack.isEmpty()) stack.remove(stack.size() - 1);
            if (stack.size() > §num§) ok = false;
        }
        return ok && stack.isEmpty();
    }
}""",
    },
    "cpp": {
        ("filter_evens", "A"): """\
#include <iostream>

int main() {
    int xs[] = {§nums§};
    int §res§[6];
    int §picked§ = 0;
    for (int i = 0; i < 6; i++)
        if (xs[i] % 2 == 0) §res§[§picked§++] = xs[i];
    std::cout << "§label§: " << §picked§ << "\\n";
    for (int i = 0; i < §picked§; i++)
        std::cout << §res§[i] << "\\n";
}""",
        ("filter_evens", "B"): """\
std::vector<int> §fn§(std::vector<int> xs) {
    std::vector<int> §res§;
    for (size_t i = 0; i < xs.size(); i++) {
        if (xs[i] % 2 == 0 && xs[i] > 0) {
            §res§.push_back(xs[i]);
        }
    }
    return §res§;
}""",
        ("word_count", "A"): """\
#include <iostream>
#include <cstring>

int main() {
    char text[] = "§s§";
    char* words[16];
    int total = 0;
    for (char* p = strtok(text, " "); p != nullptr; p = strtok(nullptr, " ")) words[total++] = p;
    int §res§ = 0;
    for (int i = 0; i < total; i++)
        for (int j = 0; j < i; j++)
            if (strcmp(words[i], words[j]) == 0) { §res§++; break; }
    std::cout << "§label§: " << §res§ << "\\n";
}""",
        ("word_count", "B"): """\
std::vector<std::string> §fn§(std::string text) {
    std::vector<std::string> words, §res§;
    std::string cur;
    text += ' ';
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] != ' ') { cur += text[i]; continue; }
        int before = 0;
        for (size_t j = 0; j < words.size(); j++)
            if (words[j] == cur) before++;
        if (before == 1) §res§.push_back(cur);
        words.push_back(cur); cur = "";
    }
    return §res§;
}""",
        ("palindrome", "A"): """\
#include <iostream>
#include <cstring>

int main() {
    char word[] = "§s§";
    int len = strlen(word);
    for (int i = 0; i < len; i++)
        if (word[i] >= 'A' && word[i] <= 'Z') word[i] = word[i] + 32;
    bool same = true;
    for (int i = 0; i < len; i++)
        if (word[i] != word[len - 1 - i]) same = false;
    if (same && len >= §num§) std::cout << "§s§: §yes§" << "\\n";
    else std::cout << "§s§: §no§" << "\\n";
}""",
        ("palindrome", "B"): """\
std::string §fn§(std::string word) {
    std::string text;
    for (size_t i = 0; i < word.size(); i++) {
        char ch = word[i];
        if (ch >= 'A' && ch <= 'Z') ch = ch + 32;
        text.push_back(ch);
    }
    std::string flipped(text.rbegin(), text.rend());
    std::string §res§ = "§no§";
    if (text == flipped && text.size() >= §num§) §res§ = "§yes§";
    return §res§;
}""",
        ("grade_stats", "A"): """\
#include <iostream>

int main() {
    int scores[] = {§nums§};
    int total = sizeof(scores) / sizeof(scores[0]);
    int §res§ = 0;
    for (int i = 0; i < total; i++) {
        if (scores[i] >= §num§) §res§ = §res§ + 1;
    }
    std::cout << "§label§: " << §res§ << " of " << total << "\\n";
    return 0;
}""",
        ("grade_stats", "B"): """\
std::vector<int> §fn§(std::vector<int> scores) {
    std::vector<int> §res§;
    for (size_t i = 0; i < scores.size(); i++) {
        if (scores[i] >= §num§) §res§.push_back(scores[i]);
    }
    return §res§;
}""",
        ("caesar", "A"): """\
#include <iostream>

int main() {
    char text[] = "§s§";
    for (int i = 0; text[i]; i++) {
        if (text[i] >= 'a' && text[i] <= 'z')
            text[i] = (text[i] - 'a' + §num§) % 26 + 'a';
    }
    std::cout << "§label§: " << text << "\\n";
    return 0;
}""",
        ("caesar", "B"): """\
std::string §fn§(std::string text) {
    std::string §res§;
    for (size_t i = 0; i < text.size(); i++) {
        char ch = text[i];
        if (ch >= 'a' && ch <= 'z') ch = (ch - 'a' + §num§) % 26 + 'a';
        §res§.push_back(ch);
    }
    return §res§;
}""",
        ("balanced_brackets", "A"): """\
#include <iostream>

int main() {
    char text[] = "§s§";
    int depth = 0;
    bool §res§ = true;
    for (int i = 0; text[i]; i++) {
        if (text[i] == '(') depth = depth + 1;
        if (text[i] == ')') depth = depth - 1;
        if (depth < 0 || depth > §num§) §res§ = false;
    }
    if (§res§ && depth == 0) std::cout << "§label§: yes" << "\\n";
    else std::cout << "§label§: no" << "\\n";
}""",
        ("balanced_brackets", "B"): """\
bool §fn§(std::string text) {
    std::vector<char> stack;
    bool deep = false;
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == '(') stack.push_back(text[i]);
        if (text[i] == ')') {
            if (stack.empty()) return false;
            stack.pop_back();
        }
        if (stack.size() > §num§) deep = true;
    }
    return stack.empty() && !deep;
}""",
    },
}
