# ===========================================================================
# One-time converter: data/2050 Investment/2050 Investment (raw export) ->
# data/2050_Investment (a flat, directly-loadable TulipaEnergyModel v0.22
# input folder).
#
# The raw export is Tulipa-shaped (asset/flow/milestone/commission tables,
# `investable`, `commission_year`, etc. all match TEM's vocabulary) but is
# missing three tables that TEM requires unconditionally (not in its
# `tables_allowed_to_be_missing` list): rep_periods_data, rep_periods_mapping,
# timeframe_data. It also stores profiles as one wide CSV
# (milestone_year, timestep, <profile_name_1>, <profile_name_2>, ...) instead
# of the long `profiles_rep_periods` table TEM reads.
#
# This is a full calendar year of hourly data (8760 timesteps) with NO
# representative-period compression, i.e. a single representative period
# that IS the year, weight 1. That's confirmed with the user, not assumed.
#
# NOTE: `TulipaIO.read_csv_folder` builds each table name from the CSV
# filename via `replace(name, "-" => "_")` - so underscore-named files (as
# this raw export already uses, e.g. `asset_milestone.csv`) map to the same
# table names as the hyphen-named files in Tulipa's own example datasets
# (`asset-milestone.csv`). No renaming is needed for the tables that already
# exist; only the missing tables + the profile reshape are new work.
#
# Usage:
#   julia --project=. convert_2050_investment.jl
#
# Output: data/2050_Investment/*.csv, ready for the same loading pattern as
# smallcase_tulipa.jl's `load_tulipa_case` (TIO.read_csv_folder + TEM
# .populate_with_defaults! + TEM.EnergyProblem).
# ===========================================================================
using DuckDB
# DuckDB re-exports DBInterface; not a direct Project.toml dep so don't `using` it.

const SRC = joinpath(@__DIR__, "data", "2050 Investment", "2050 Investment")
const DST = joinpath(@__DIR__, "data", "2050_Investment")

isdir(SRC) || error("source folder not found: $SRC")
mkpath(DST)

println("[convert] source: $SRC")
println("[convert] dest:   $DST")

# --- 1. copy every existing table's CSV through unchanged (except the wide
#        profile table, which gets reshaped below) --------------------------
ncopied = 0
for f in readdir(SRC)
    endswith(f, ".csv") || continue
    f == "profiles_wide.csv" && continue
    cp(joinpath(SRC, f), joinpath(DST, f); force = true)
    global ncopied += 1
end
println("[convert] copied $ncopied table CSVs unchanged")

# --- 2. reshape profiles_wide.csv (wide) -> profiles_rep_periods.csv (long) -
con = DBInterface.connect(DuckDB.DB)
wide_path = joinpath(SRC, "profiles_wide.csv")
DuckDB.query(con, "CREATE TABLE wide AS SELECT * FROM read_csv_auto('$wide_path')")

cols = [row.name for row in DuckDB.query(con, "PRAGMA table_info('wide')")]
profile_cols = filter(c -> !(c in ("milestone_year", "timestep")), cols)
isempty(profile_cols) && error("no profile columns found in profiles_wide.csv")

unpivot_list = join("\"" .* profile_cols .* "\"", ", ")
out_path = joinpath(DST, "profiles_rep_periods.csv")
DuckDB.query(
    con,
    """
    COPY (
        SELECT
            milestone_year,
            profile_name,
            1 AS rep_period,
            timestep,
            value
        FROM wide
        UNPIVOT (value FOR profile_name IN ($unpivot_list))
    ) TO '$out_path' (HEADER, DELIMITER ',')
    """,
)
nrows = only(DuckDB.query(con, "SELECT count(*) AS n FROM read_csv_auto('$out_path')")).n
println(
    "[convert] wrote profiles_rep_periods.csv ($nrows rows, $(length(profile_cols)) profiles)",
)

# --- 3. synthesize the missing required tables ------------------------------
# milestone_year is 2050 throughout the raw export (model_parameters,
# asset_milestone, ...); num_timesteps/resolution come straight from the
# 8760-row wide profile table (hourly, full year, single rep period).
milestone_year =
    only(DuckDB.query(con, "SELECT DISTINCT milestone_year FROM wide")).milestone_year
num_timesteps = only(DuckDB.query(con, "SELECT max(timestep) AS n FROM wide")).n
println("[convert] milestone_year=$milestone_year  num_timesteps=$num_timesteps")

open(joinpath(DST, "rep_periods_data.csv"), "w") do io
    println(io, "milestone_year,rep_period,num_timesteps,resolution")
    println(io, "$milestone_year,1,$num_timesteps,1.0")
end

open(joinpath(DST, "rep_periods_mapping.csv"), "w") do io
    println(io, "milestone_year,period,rep_period,scenario,weight")
    println(io, "$milestone_year,1,1,1,1.0")
end

open(joinpath(DST, "timeframe_data.csv"), "w") do io
    println(io, "milestone_year,period,num_timesteps")
    println(io, "$milestone_year,1,$num_timesteps")
end
println("[convert] wrote rep_periods_data.csv, rep_periods_mapping.csv, timeframe_data.csv")

DBInterface.close!(con)
println("[convert] DONE -> $DST")
