"""Render every remote branch + merge of this repo as a GitKraken-style railway SVG.

Time runs down; `main` is the thick lane pinned at the left; every other branch
gets a lane coloured by owner. Real merges are solid curves into hollow merge
nodes; squash/rebase PRs (recovered from GitHub via `gh`) land on diamonds, with
a dashed curve back to the pre-squash tip when that branch still exists and grey
stubs when it has been deleted. Branch heads are labelled in the right gutter
with their fate (merged into main / squash-merged / folded into another branch /
open PR / closed never merged / dangling).

    uv run python scripts/render_branch_graph.py            # -> docs/wiki/assets/branch-spaghetti.svg
    uv run python scripts/render_branch_graph.py --out x.svg --no-gh

Reads only remote-tracking refs (`origin/*`), so run `git fetch --prune origin`
first for an up-to-date picture. `gh` is optional: without it the PR overlays
(squash landings, open/closed status, arch-fleet PR counts) are omitted.
Stdlib only; the owner palette below is the dataviz reference palette, validated
for CVD separation (the 5th hue was checked separately).
"""
import argparse
import html
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------- repo-specific: who owns which branches
PEOPLE = [  # (owner, branch-prefix regex, author regex) — prefix wins, author is the fallback
    ('Jonathan',   re.compile(r'^(jb|jonathan)/'), re.compile(r'jonathan|bostock', re.IGNORECASE)),
    ('Sid',        re.compile(r'^sid/'),           re.compile(r'\bsid\b|sidbaines', re.IGNORECASE)),
    ('Angel',      re.compile(r'^am/'),            re.compile(r'angel|ma-rmartinez', re.IGNORECASE)),
    ('Daniel',     re.compile(r'^dt/'),            re.compile(r'dtch1997|daniel tan', re.IGNORECASE)),
    ('arch fleet', re.compile(r'^arch/'),          re.compile(r'arch.?worker|worker@|arch@', re.IGNORECASE)),
]
COLOR = {'main': '#0b0b0b', 'Jonathan': '#2a78d6', 'Sid': '#eb6834', 'Angel': '#1baf7a',
         'Daniel': '#4a3aa7', 'arch fleet': '#c2185b', 'other': '#9a9891'}

# ---------------------------------------------------------------- geometry + surface
ROW_H, LANE_W = 5.0, 14.0
X0, TOP = 78.0, 184.0
R_NODE, R_MERGE = 2.0, 2.9
FONT = 9.0
LABEL_LH, LABEL_W = FONT + 2.0, 500
SURFACE, GRID = '#fcfcfb', '#e6e5e0'
TEXT1, TEXT2, TEXT3 = '#0b0b0b', '#52514e', '#8a887f'


def sh(*cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def owner_of(branch, author):
    if branch:
        for name, pre, _ in PEOPLE:
            if pre.search(branch):
                return name
    for name, _, au in PEOPLE:
        if au.search(author or ''):
            return name
    return 'other'


def parse_merge_branch(subj):
    m = re.search(r"Merge pull request #\d+ from [^/\s]+/(\S+)", subj)
    if m: return m.group(1)
    m = re.search(r"'(?:origin/)?([^']+)'", subj)
    if m: return m.group(1)
    m = re.search(r"(?<![\w@'/])(?:origin/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_./-]*[A-Za-z0-9_])", subj)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--repo', default=None, help='repo root (default: this script\'s repo)')
    ap.add_argument('--out', default=None, help='output SVG (default: docs/wiki/assets/branch-spaghetti.svg)')
    ap.add_argument('--main', default='origin/main', help='trunk ref')
    ap.add_argument('--no-gh', action='store_true', help='skip GitHub PR overlays')
    ap.add_argument('--alloc', choices=['right', 'left'], default='right',
                    help='lane allocation for merged-in branches: nest right of the parent, or leftmost free')
    args = ap.parse_args()
    repo = Path(args.repo) if args.repo else Path(sh('git', 'rev-parse', '--show-toplevel', cwd=Path(__file__).resolve().parent).strip())
    out = Path(args.out) if args.out else repo / 'docs/wiki/assets/branch-spaghetti.svg'
    repo_name = repo.name

    # ------------------------------------------------------------ data
    commits = []
    for line in sh('git', 'log', '--remotes', '--date-order', '--format=%H|%P|%ct|%an|%ae|%s', cwd=repo).splitlines():
        h, p, ct, an, ae, s = line.split('|', 5)
        commits.append({'h': h, 'parents': p.split() if p else [], 't': int(ct), 'an': an, 'ae': ae, 's': s})
    row = {c['h']: i for i, c in enumerate(commits)}
    by = {c['h']: c for c in commits}
    N = len(commits)
    in_main = set(sh('git', 'rev-list', args.main, cwd=repo).split())
    main_short = args.main.split('/', 1)[-1]

    refs = defaultdict(set)
    for line in sh('git', 'for-each-ref', '--format=%(objectname) %(refname:short)', 'refs/remotes', cwd=repo).splitlines():
        h, name = line.split()
        if name in ('origin', 'origin/HEAD'):
            continue
        refs[h].add(name.split('/', 1)[1] if name.startswith('origin/') else name)
    refnames = {h: sorted(ns, key=lambda x: (x != main_short, x)) for h, ns in refs.items() if h in row}
    MAIN_TIP = next(h for h, ns in refnames.items() if main_short in ns)

    prs = []
    if not args.no_gh:
        try:
            prs = json.loads(sh('gh', 'pr', 'list', '--state', 'all', '--limit', '1000', '--json',
                                'number,headRefName,headRefOid,baseRefName,state,mergeCommit', cwd=repo))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f'warning: gh unavailable ({e}); rendering without PR overlays', file=sys.stderr)
    pr_by_head, prs_by_base = defaultdict(list), defaultdict(list)
    for p in prs:
        pr_by_head[p['headRefName']].append(p); prs_by_base[p['baseRefName']].append(p)
    STATE_RANK = {'MERGED': 0, 'OPEN': 1, 'CLOSED': 2}
    def best_pr(names):
        ps = [p for n in names for p in pr_by_head.get(n, [])]
        return min(ps, key=lambda p: (STATE_RANK[p['state']], -p['number'])) if ps else None

    # ------------------------------------------------------------ lane layout (rows newest-first; a lane = a pending first-parent line)
    lanes = [{'expect': MAIN_TIP, 'color': COLOR['main'], 'owner': 'main'}]   # lane 0 reserved for the trunk
    def lane_free(i, r):
        L = lanes[i]
        return L is None or ('cool' in L and L['cool'] + 1 < r)
    def alloc_lane(r, right_of=None):
        rng = range(right_of + 1, len(lanes)) if right_of is not None else range(len(lanes))
        for i in rng:
            if lane_free(i, r): return i
        lanes.append(None); return len(lanes) - 1
    def find_pending(h):
        for i, L in enumerate(lanes):
            if L and L.get('expect') == h: return i
        return None

    node_lane, node_color, node_owner = {}, {}, {}
    edges, is_tip, occupancy = [], set(), []
    for r, c in enumerate(commits):
        h = c['h']; author = f"{c['an']} {c['ae']}"
        expecting = [i for i, L in enumerate(lanes) if L and L.get('expect') == h]
        if expecting:
            my = expecting[0]; color, owner = lanes[my]['color'], lanes[my]['owner']
            for i in expecting[1:]: lanes[i] = {'cool': r}
        else:
            is_tip.add(h)
            names = refnames.get(h, []); bname = names[0] if names else None
            owner = 'main' if bname == main_short else owner_of(bname, author)
            color = COLOR[owner]; my = alloc_lane(r)
        node_lane[h], node_color[h], node_owner[h] = my, color, owner
        lanes[my] = {'cool': r}
        for k, p in enumerate(c['parents']):
            j = find_pending(p)
            if k == 0:
                if j is not None and j < my:
                    edges.append((h, p, j, lanes[j]['color']))
                else:  # keep our own lane -> stable trunk, stable long-lived branches
                    lanes[my] = {'expect': p, 'color': color, 'owner': owner}
                    edges.append((h, p, my, color))
            elif j is not None:
                edges.append((h, p, j, lanes[j]['color']))
            else:
                pc = by.get(p)
                own2 = owner_of(parse_merge_branch(c['s']), f"{pc['an']} {pc['ae']}" if pc else '')
                j = alloc_lane(r, right_of=my if args.alloc == 'right' else None)
                lanes[j] = {'expect': p, 'color': COLOR[own2], 'owner': own2}
                edges.append((h, p, j, COLOR[own2]))
        occupancy.append(sum(1 for L in lanes if L and 'expect' in L))
    max_lanes = len(lanes)

    # ------------------------------------------------------------ PR overlays
    phantoms, ghost_landings, true_merge_prs, unreachable_prs = [], Counter(), 0, 0
    for p in prs:
        if p['state'] != 'MERGED' or not p.get('mergeCommit'):
            continue
        M, H = p['mergeCommit']['oid'], p.get('headRefOid')
        if M not in row: unreachable_prs += 1; continue
        if H in by[M]['parents']: true_merge_prs += 1
        elif H in row: phantoms.append((H, M, node_color[H], p['number']))
        else: ghost_landings[M] += 1
    landing = set(ghost_landings) | {M for _, M, _, _ in phantoms}

    def ref_status(h):
        names = refnames[h]
        if h == MAIN_TIP: return 'trunk', ''
        pr = best_pr(names)
        if h not in is_tip:
            if h in in_main:
                return 'merged', f'merged into {main_short}  #{pr["number"]}' if pr and pr['state'] == 'MERGED' else f'merged into {main_short}'
            return 'folded', 'folded into another branch'
        if pr is None: return 'nopr', 'dangling, no PR'
        if pr['state'] == 'MERGED': return 'squash', f'squash-merged  #{pr["number"]}'
        if pr['state'] == 'OPEN': return 'open', f'PR #{pr["number"]} open'
        return 'closed', f'PR #{pr["number"]} closed, never merged'

    # ------------------------------------------------------------ SVG
    def x(l): return X0 + l * LANE_W
    def y(r): return TOP + r * ROW_H
    def esc(s): return html.escape(s, quote=True)
    def edge_path(ch, pa, j):
        xc, yc, xp, yp, xj = x(node_lane[ch]), y(row[ch]), x(node_lane[pa]), y(row[pa]), x(j)
        if row[pa] - row[ch] == 1:
            if xc == xp: return f'M{xc:.1f},{yc:.1f}V{yp:.1f}'
            return f'M{xc:.1f},{yc:.1f}C{xc:.1f},{yc+ROW_H*.55:.1f} {xp:.1f},{yp-ROW_H*.55:.1f} {xp:.1f},{yp:.1f}'
        d = f'M{xc:.1f},{yc:.1f}'
        if xj != xc: d += f'C{xc:.1f},{yc+ROW_H*.6:.1f} {xj:.1f},{yc+ROW_H*.4:.1f} {xj:.1f},{yc+ROW_H:.1f}'
        if xj != xp: d += f'V{yp-ROW_H:.1f}C{xj:.1f},{yp-ROW_H*.4:.1f} {xp:.1f},{yp-ROW_H*.6:.1f} {xp:.1f},{yp:.1f}'
        else: d += f'V{yp:.1f}'
        return d

    X_LABEL = X0 + max_lanes * LANE_W + 26
    WIDTH = max(X_LABEL + LABEL_W, 1000)
    HEIGHT = y(N) + 30
    o = []; A = o.append
    A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH:.0f}" height="{HEIGHT:.0f}" viewBox="0 0 {WIDTH:.0f} {HEIGHT:.0f}" font-family="Inter, Helvetica, Arial, sans-serif">')
    A(f'<rect width="100%" height="100%" fill="{SURFACE}"/>')

    A(f'<g id="weeks" stroke="{GRID}" stroke-width="1" font-size="{FONT}" fill="{TEXT2}">')
    prev = None
    for r, c in enumerate(commits):
        d = datetime.fromtimestamp(c['t'], tz=UTC); wk = d.isocalendar()[:2]
        if wk != prev:
            monday = d - timedelta(days=d.weekday()); yy = y(r) - ROW_H / 2
            A(f'<line x1="{X0-8:.1f}" x2="{X_LABEL-10:.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            A(f'<text x="{X0-12:.1f}" y="{yy+FONT:.1f}" text-anchor="end" stroke="none">{monday.strftime("%b %d")}</text>')
            prev = wk
    A('</g>')

    A('<g id="edges" fill="none" stroke-width="1.4" stroke-linecap="round">')
    for ch, pa, j, col in edges:
        if pa not in row: continue
        trunk = node_owner[ch] == 'main' and node_owner[pa] == 'main' and j == 0
        width = ' stroke-width="2.4"' if trunk else ''
        A(f'<path d="{edge_path(ch, pa, j)}" stroke="{col}"{width}/>')
    A('</g>')

    A('<g id="squash" fill="none" stroke-width="1.1" stroke-dasharray="3 2.5" opacity="0.75">')
    for H, M, col, num in phantoms:
        xh, yh, xm, ym = x(node_lane[H]), y(row[H]), x(node_lane[M]), y(row[M])
        A(f'<path d="M{xh:.1f},{yh:.1f}C{xh:.1f},{(yh+ym)/2:.1f} {xm:.1f},{(yh+ym)/2:.1f} {xm:.1f},{ym:.1f}" stroke="{col}"><title>PR #{num}: squash-merged</title></path>')
    A('</g>')

    A(f'<g id="ghosts" fill="none" stroke="{TEXT3}" stroke-width="1" stroke-dasharray="2 2">')
    for M, n in ghost_landings.items():
        xm, ym = x(node_lane[M]), y(row[M])
        for i in range(min(n, 4)):
            A(f'<path d="M{xm:.1f},{ym:.1f}C{xm+LANE_W*0.5:.1f},{ym:.1f} {xm+LANE_W*0.7:.1f},{ym-ROW_H*(1.2+i*0.8):.1f} {xm+LANE_W*0.9+i*2:.1f},{ym-ROW_H*(2.6+i*1.2):.1f}"><title>{n} squash-merged PR(s) landed here; source branch deleted</title></path>')
    A('</g>')

    A(f'<g id="nodes" stroke="{SURFACE}" stroke-width="0.9">')
    for c in commits:
        h = c['h']; cx, cy = x(node_lane[h]), y(row[h]); col = node_color[h]
        title = esc(f"{h[:8]}  {datetime.fromtimestamp(c['t'], tz=UTC):%Y-%m-%d}  {c['an']}\n{c['s']}")
        if h in landing:
            s = R_MERGE + 0.7
            A(f'<path d="M{cx:.1f},{cy-s:.1f}L{cx+s:.1f},{cy:.1f}L{cx:.1f},{cy+s:.1f}L{cx-s:.1f},{cy:.1f}Z" fill="{col}"><title>{title}</title></path>')
        elif len(c['parents']) > 1:
            A(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R_MERGE}" fill="{SURFACE}" stroke="{col}" stroke-width="1.3"><title>{title}</title></circle>')
        else:
            A(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R_NODE}" fill="{col}"><title>{title}</title></circle>')
    A('</g>')

    labels = sorted((row[h], h) for h in refnames)
    A(f'<g id="labels" font-size="{FONT}">')
    ylab_prev, status_counts = -1e9, Counter()
    for r, h in labels:
        kind, note = ref_status(h); status_counts[kind] += 1
        yl = max(y(r), ylab_prev + LABEL_LH); ylab_prev = yl
        cx, cy, col = x(node_lane[h]), y(r), node_color[h]
        A(f'<path d="M{cx:.1f},{cy:.1f}H{X_LABEL-14:.1f}L{X_LABEL-6:.1f},{yl:.1f}" fill="none" stroke="{col}" stroke-width="0.7" opacity="0.45"/>')
        gx = X_LABEL - 0.4
        if kind == 'closed':
            A(f'<path d="M{gx-2.8:.1f},{yl-2.8:.1f}l5.6,5.6m0,-5.6l-5.6,5.6" stroke="{COLOR["other"]}" stroke-width="1.3" fill="none"/>')
        elif kind == 'open':
            A(f'<circle cx="{gx:.1f}" cy="{yl:.1f}" r="2.7" fill="none" stroke="{col}" stroke-width="1.2"/>')
        elif kind == 'trunk':
            A(f'<rect x="{gx-3:.1f}" y="{yl-3:.1f}" width="6" height="6" fill="{col}"/>')
        else:
            A(f'<circle cx="{gx:.1f}" cy="{yl:.1f}" r="2.4" fill="{col}"/>')
        names = refnames[h]; base_note = ''
        for n in names:
            if len(prs_by_base.get(n, [])) >= 3:
                cnt = Counter(p['state'] for p in prs_by_base[n])
                base_note = f'· target of {len(prs_by_base[n])} PRs ({cnt["MERGED"]} merged / {cnt["CLOSED"]} closed / {cnt["OPEN"]} open)'; break
        A(f'<text x="{X_LABEL+7:.1f}" y="{yl+FONT*0.36:.1f}" fill="{TEXT1}">{esc(", ".join(names))}'
          + (f'<tspan fill="{TEXT3}" dx="6">{esc(note)}</tspan>' if note else '')
          + (f'<tspan fill="{TEXT2}" dx="6">{esc(base_note)}</tspan>' if base_note else '') + '</text>')
    A('</g>')

    first = datetime.fromtimestamp(commits[-1]['t'], tz=UTC); last = datetime.fromtimestamp(commits[0]['t'], tz=UTC)
    n_merge_commits = sum(1 for c in commits if len(c['parents']) > 1)
    n_names = len({n for ns in refnames.values() for n in ns})
    pr_states = Counter(p['state'] for p in prs)
    hx = X0 - 8
    A('<g id="header">')
    A(f'<text x="{hx:.0f}" y="32" font-size="20" font-weight="600" fill="{TEXT1}">{esc(repo_name)} — every branch, every merge</text>')
    A(f'<text x="{hx:.0f}" y="52" font-size="11" fill="{TEXT2}">{first:%d %b} → {last:%d %b %Y} · {N:,} commits, {N-len(in_main):,} of them not reachable from {main_short} · '
      f'{n_names} branches on {len(labels)} distinct heads · {n_merge_commits} merge commits · peak {max(occupancy)} branches alive at once</text>')
    if prs:
        A(f'<text x="{hx:.0f}" y="68" font-size="11" fill="{TEXT2}">{len(prs)} PRs · {pr_states["MERGED"]} merged: {true_merge_prs} true merges, {len(phantoms)} squash/rebase with the tip still around, '
          f'{sum(ghost_landings.values())} squash/rebase with the branch since deleted, {unreachable_prs} into branches since deleted</text>')
        A(f'<text x="{hx:.0f}" y="84" font-size="11" fill="{TEXT2}">{pr_states["CLOSED"]} PRs closed without merging · {pr_states["OPEN"]} still open</text>')
    def legend_row(yy, caption, items):
        lx = hx
        A(f'<text x="{lx:.0f}" y="{yy}" font-size="10" fill="{TEXT2}">{caption}</text>'); lx += 158
        for glyph, label in items:
            A(glyph.format(x=lx + 8, y=yy - 3.5, xl=lx, xr=lx + 16))
            A(f'<text x="{lx+22}" y="{yy}" font-size="10" fill="{TEXT1}">{esc(label)}</text>')
            lx += 22 + len(label) * 5.3 + 16
    legend_row(110, 'lane colour = branch owner', [(f'<line x1="{{xl}}" x2="{{xr}}" y1="{{y}}" y2="{{y}}" stroke="{COLOR[o_]}" stroke-width="2.4"/>', o_)
                                                   for o_ in COLOR])
    legend_row(132, 'marks', [
        (f'<circle cx="{{x}}" cy="{{y}}" r="2.4" fill="{TEXT1}"/>', 'commit'),
        (f'<circle cx="{{x}}" cy="{{y}}" r="3" fill="none" stroke="{TEXT1}" stroke-width="1.3"/>', 'merge commit'),
        (f'<path d="M{{x}},{{y}}m0,-3.6l3.6,3.6l-3.6,3.6l-3.6,-3.6z" fill="{TEXT1}"/>', 'squash-merge landing'),
        (f'<line x1="{{xl}}" x2="{{xr}}" y1="{{y}}" y2="{{y}}" stroke="{COLOR["Sid"]}" stroke-width="1.2" stroke-dasharray="3 2.5"/>', 'squash/rebase PR (tip → landing)'),
        (f'<line x1="{{xl}}" x2="{{xr}}" y1="{{y}}" y2="{{y}}" stroke="{TEXT3}" stroke-width="1" stroke-dasharray="2 2"/>', 'squash-merged PR, branch deleted'),
    ])
    legend_row(154, 'branch head (right gutter)', [
        (f'<rect x="{{x}}" y="{{y}}" width="6" height="6" transform="translate(-3,-3)" fill="{TEXT1}"/>', main_short),
        (f'<circle cx="{{x}}" cy="{{y}}" r="2.4" fill="{TEXT1}"/>', 'merged / squash-merged / folded / dangling (see note)'),
        (f'<circle cx="{{x}}" cy="{{y}}" r="2.7" fill="none" stroke="{TEXT1}" stroke-width="1.2"/>', 'open PR'),
        (f'<path d="M{{x}},{{y}}m-2.8,-2.8l5.6,5.6m0,-5.6l-5.6,5.6" stroke="{COLOR["other"]}" stroke-width="1.3" fill="none"/>', 'PR closed, never merged'),
    ])
    A('</g></svg>')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(o))

    pk = max(range(N), key=lambda i: occupancy[i])
    print(f'wrote {out} ({WIDTH:.0f}x{HEIGHT:.0f}px): {N} commits, {N-len(in_main)} not in {main_short}, '
          f'{n_names} branches / {len(labels)} heads, {n_merge_commits} merge commits, peak {max(occupancy)} lanes on '
          f'{datetime.fromtimestamp(commits[pk]["t"], tz=UTC):%Y-%m-%d}')
    print('heads by status:', dict(status_counts))
    if prs:
        print(f'PRs: {dict(pr_states)}; merged = {true_merge_prs} true + {len(phantoms)} squash(tip alive) + '
              f'{sum(ghost_landings.values())} squash(branch gone) + {unreachable_prs} into deleted bases')


if __name__ == '__main__':
    main()
