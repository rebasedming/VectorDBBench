#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_pg_vector_search_perf.sh DB_URL [options]

Runs a single Postgres backend in a tight vector-search loop and attaches Linux
perf to that backend PID. Intended for profiling a VectorDBBench-created Cohere
1M pg_search table without profiling VectorDBBench/Python runner overhead.

Options:
  --table NAME             Table name, optionally schema-qualified
                           (default: public.vdbbench_pg_search)
  --id-column NAME         Primary key column (default: id)
  --vector-column NAME     Vector column (default: embedding)
  --metric cosine|l2|ip    Distance operator: cosine=<=>, l2=<->, ip=<#>
                           (default: cosine)
  --operator OP            Override distance operator directly
  --k N                    LIMIT value (default: 100)
  --probes N               paradedb.vector_cluster_probes (default: 50)
  --rerank FLOAT           paradedb.vector_rerank_multiplier (default: 1.0)
  --where SQL              Extra SQL appended after paradedb.all(), e.g.
                           "AND id >= 10000" or "AND label = 'x'"
  --test-parquet PATH      Parquet file containing the query vectors
                           (default: $DATASET_LOCAL_DIR/cohere/cohere_medium_1m/test.parquet,
                           with DATASET_LOCAL_DIR defaulting to /tmp/vectordb_bench/dataset)
  --query-index N          Row in test parquet to use (default: 0)
  --vector-literal '[...]' Use this vector literal instead of reading parquet
  --duration SECONDS       perf recording duration (default: 30)
  --frequency HZ           perf sample frequency (default: 99)
  --call-graph MODE        perf call graph mode: dwarf, fp, or lbr (default: dwarf)
  --output PATH            perf.data output path (default: perf-pg-vector.data)
  --start-delay SECONDS    Backend sleep before loop, gives perf time to attach
                           (default: 5)
  --no-sudo                Run perf without sudo
  -h, --help               Show this help

Examples:
  scripts/profile_pg_vector_search_perf.sh "$DATABASE_URL" --duration 60

  scripts/profile_pg_vector_search_perf.sh "$DATABASE_URL" \
    --table public.vdbbench_pg_search --probes 150 --rerank 1.0 \
    --where "AND id >= 10000" --output pgsearch-id-filter.perf.data
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

quote_ident() {
  local ident=$1
  [[ $ident =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "invalid SQL identifier: $ident"
  printf '"%s"' "$ident"
}

quote_qualified_ident() {
  local name=$1
  local IFS=.
  local parts=($name)
  [[ ${#parts[@]} -ge 1 && ${#parts[@]} -le 2 ]] || die "invalid table name: $name"
  if [[ ${#parts[@]} -eq 1 ]]; then
    quote_ident "${parts[0]}"
  else
    printf '%s.%s' "$(quote_ident "${parts[0]}")" "$(quote_ident "${parts[1]}")"
  fi
}

db_url=${1:-}
if [[ -z ${db_url} || ${db_url} == "-h" || ${db_url} == "--help" ]]; then
  usage
  [[ -z ${db_url} ]] && exit 1 || exit 0
fi
shift

table="public.vdbbench_pg_search"
id_column="id"
vector_column="embedding"
metric="cosine"
operator=""
k=100
probes=50
rerank="1.0"
extra_where=""
dataset_root="${DATASET_LOCAL_DIR:-/tmp/vectordb_bench/dataset}"
test_parquet=""
query_index=0
vector_literal=""
duration=30
frequency=99
call_graph="dwarf"
output="perf-pg-vector.data"
start_delay=5
use_sudo=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --table) table=${2:?}; shift 2 ;;
    --id-column) id_column=${2:?}; shift 2 ;;
    --vector-column) vector_column=${2:?}; shift 2 ;;
    --metric) metric=${2:?}; shift 2 ;;
    --operator) operator=${2:?}; shift 2 ;;
    --k) k=${2:?}; shift 2 ;;
    --probes) probes=${2:?}; shift 2 ;;
    --rerank) rerank=${2:?}; shift 2 ;;
    --where) extra_where=${2:?}; shift 2 ;;
    --test-parquet) test_parquet=${2:?}; shift 2 ;;
    --query-index) query_index=${2:?}; shift 2 ;;
    --vector-literal) vector_literal=${2:?}; shift 2 ;;
    --duration) duration=${2:?}; shift 2 ;;
    --frequency) frequency=${2:?}; shift 2 ;;
    --call-graph) call_graph=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --start-delay) start_delay=${2:?}; shift 2 ;;
    --no-sudo) use_sudo=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ $k =~ ^[0-9]+$ ]] || die "--k must be an integer"
[[ $probes =~ ^[0-9]+$ ]] || die "--probes must be an integer"
[[ $query_index =~ ^[0-9]+$ ]] || die "--query-index must be an integer"
[[ $duration =~ ^[0-9]+$ ]] || die "--duration must be an integer"
[[ $frequency =~ ^[0-9]+$ ]] || die "--frequency must be an integer"
[[ $start_delay =~ ^[0-9]+$ ]] || die "--start-delay must be an integer"

if [[ -z $operator ]]; then
  case "$metric" in
    cosine) operator="<=>" ;;
    l2) operator="<->" ;;
    ip) operator="<#>" ;;
    *) die "--metric must be one of: cosine, l2, ip" ;;
  esac
fi
case "$operator" in
  "<=>"|"<->"|"<#>") ;;
  *) die "--operator must be one of: <=>, <->, <#>" ;;
esac
case "$call_graph" in
  dwarf|fp|lbr) ;;
  *) die "--call-graph must be one of: dwarf, fp, lbr" ;;
esac

require_cmd psql
require_cmd perf
require_cmd python3

if [[ -z $vector_literal ]]; then
  if [[ -z $test_parquet ]]; then
    test_parquet="${dataset_root}/cohere/cohere_medium_1m/test.parquet"
  fi
  [[ -f $test_parquet ]] || die "test parquet not found: $test_parquet"
  vector_literal=$(
    python3 - "$test_parquet" "$query_index" <<'PY'
import sys

import polars as pl

path = sys.argv[1]
idx = int(sys.argv[2])
df = pl.read_parquet(path, columns=["emb"], n_rows=idx + 1)
if idx >= df.height:
    raise SystemExit(f"query index {idx} out of range for {path}")
vec = df["emb"][idx]
print("[" + ",".join(str(float(x)) for x in vec) + "]")
PY
  )
fi

[[ $vector_literal == \[*\] ]] || die "vector literal must look like '[1.0,2.0,...]'"

table_sql=$(quote_qualified_ident "$table")
id_sql=$(quote_ident "$id_column")
vector_sql=$(quote_ident "$vector_column")

tmpdir=$(mktemp -d)
sql_file="${tmpdir}/loop.sql"
pid_file="${tmpdir}/backend.pid"
psql_log="${tmpdir}/psql.log"
psql_pid=""

cleanup() {
  if [[ -n ${psql_pid} ]] && kill -0 "$psql_pid" >/dev/null 2>&1; then
    kill "$psql_pid" >/dev/null 2>&1 || true
    wait "$psql_pid" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmpdir"
}
trap cleanup EXIT INT TERM

cat >"$sql_file" <<SQL
\\set ON_ERROR_STOP on
SET client_min_messages = warning;
SET statement_timeout = 0;
SET max_parallel_workers_per_gather = 0;
SET paradedb.vector_cluster_probes = ${probes};
SET paradedb.vector_rerank_multiplier = ${rerank};
SELECT pg_backend_pid();
SELECT pg_sleep(${start_delay});
DO \$profile_loop\$
BEGIN
  LOOP
    PERFORM ${id_sql}
    FROM ${table_sql}
    WHERE ${id_sql} @@@ paradedb.all()
      ${extra_where}
    ORDER BY ${vector_sql} ${operator} '${vector_literal}'::vector
    LIMIT ${k};
  END LOOP;
END
\$profile_loop\$;
SQL

echo "starting query backend..."
psql "$db_url" -X -qAt -f "$sql_file" >"$pid_file" 2>"$psql_log" &
psql_pid=$!

backend_pid=""
deadline=$((SECONDS + start_delay))
while [[ $SECONDS -le $deadline ]]; do
  if [[ -s $pid_file ]]; then
    backend_pid=$(sed -n 's/^\([0-9][0-9]*\)$/\1/p' "$pid_file" | head -n 1)
    [[ -n $backend_pid ]] && break
  fi
  if ! kill -0 "$psql_pid" >/dev/null 2>&1; then
    echo "psql failed; stderr:" >&2
    cat "$psql_log" >&2
    exit 1
  fi
  sleep 0.1
done

[[ -n $backend_pid ]] || die "could not capture backend PID; psql stderr is in $psql_log"

perf_cmd=(perf record -F "$frequency" -g --call-graph "$call_graph" -p "$backend_pid" -o "$output" -- sleep "$duration")
if [[ $use_sudo -eq 1 && $EUID -ne 0 ]]; then
  require_cmd sudo
  echo "requesting sudo for perf..."
  sudo -v
  perf_cmd=(sudo "${perf_cmd[@]}")
fi

echo "backend PID: $backend_pid"
echo "recording perf for ${duration}s -> $output"
"${perf_cmd[@]}"

echo "done."
echo "Inspect with: perf report -i '$output'"
echo "Or export stacks with: perf script -i '$output'"
