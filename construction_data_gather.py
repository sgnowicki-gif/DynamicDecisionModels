import requests
import pandas as pd
import ast

NDurl = 'https://travelfiles.dot.nd.gov/geojson_nc/wzdx_geojson.json'
MIurl = 'https://mn.carsprogram.org/carsapi_v1/api/wzdx'
WIurl = 'https://511wi.gov/api/wzdx'
IAurl = 'https://iowa-atms.cloud-q-free.com/api/rest/dataprism/wzdx/wzdxfeed'

def saveDf(state, state_URL):
    data = requests.get(state_URL).json()

    rows = [feature['properties'] for feature in data['features']]
    
    df = pd.DataFrame(rows)

    df.to_csv(f'{state}_construction.csv')

    print(f"Saved {len(df)} records")

import pandas as pd
import ast


def build_project_database(
    files,
    output_file="combined_projects.csv",
    min_duration_days=None,
    remove_short_term_events=False
):
    """
    Combine MI, ND, WI, and IA WZDx CSV exports into a single
    project-level dataset.

    Example:
    ----------
    projects = build_project_database(
        {
            "MI": "MI_construction.csv",
            "ND": "ND_construction.csv",
            "WI": "WI_construction.csv",
            "IA": "IO_construction.csv"
        },
        output_file="combined_projects.csv",
        min_duration_days=7,
        remove_short_term_events=True
    )

    Output fields:
    ----------
    state
    project_id
    project_name
    road
    direction
    project_type
    description
    start_date
    end_date
    duration_days
    vehicle_impact
    begin_location
    end_location
    """

    all_projects = []

    for state, filename in files.items():

        print(f"Reading {state}: {filename}")

        df = pd.read_csv(filename)

        for _, row in df.iterrows():

            # ----------------------------------
            # Parse core_details dictionary
            # ----------------------------------

            details = {}

            if "core_details" in df.columns:

                try:
                    details = ast.literal_eval(str(row["core_details"]))
                except:
                    details = {}

            # ----------------------------------
            # Skip detour events
            # ----------------------------------

            if details.get("event_type") == "detour":
                continue

            # ----------------------------------
            # Road
            # ----------------------------------

            road = None

            road_names = details.get("road_names", [])

            if isinstance(road_names, list) and len(road_names):
                road = road_names[0]

            # ----------------------------------
            # Basic attributes
            # ----------------------------------

            direction = details.get("direction")
            description = details.get("description")
            event_name = details.get("name")

            project_name = (
                event_name
                if event_name
                else description
            )

            # ----------------------------------
            # Project ID
            # ----------------------------------

            if event_name:

                # Wisconsin / Iowa style IDs
                project_id = f"{state}_{event_name}"

            else:

                # Michigan / North Dakota fallback
                project_id = (
                    f"{state}_"
                    f"{road}_"
                    f"{str(description)[:100]}"
                )

            # ----------------------------------
            # Project type
            # ----------------------------------

            project_type = None

            if "types_of_work" in row.index:

                try:

                    work = ast.literal_eval(
                        str(row["types_of_work"])
                    )

                    if isinstance(work, list):

                        type_names = []

                        for item in work:

                            if isinstance(item, dict):

                                if "type_name" in item:
                                    type_names.append(
                                        item["type_name"]
                                    )

                        if len(type_names):
                            project_type = ",".join(type_names)

                except:
                    pass

            if not project_type:

                desc = str(description).lower()

                if "bridge" in desc:
                    project_type = "bridge"

                elif "construction" in desc:
                    project_type = "construction"

                elif "maintenance" in desc:
                    project_type = "maintenance"

                elif ("surface" in desc or
                      "pavement" in desc or
                      "paving" in desc):
                    project_type = "surface"

                else:
                    project_type = "other"

            # ----------------------------------
            # Dates
            # ----------------------------------

            start = pd.to_datetime(
                row.get("start_date"),
                errors="coerce"
            )

            end = pd.to_datetime(
                row.get("end_date"),
                errors="coerce"
            )

            duration_days = None

            if pd.notna(start) and pd.notna(end):

                duration_days = (
                    end - start
                ).total_seconds() / 86400

            # ----------------------------------
            # Vehicle impact
            # ----------------------------------

            vehicle_impact = row.get(
                "vehicle_impact",
                None
            )

            # ----------------------------------
            # Beginning location
            # ----------------------------------

            begin_loc = None

            if "beginning_cross_street" in row.index:

                begin_loc = row.get(
                    "beginning_cross_street"
                )

            elif "beginning_milepost" in row.index:

                begin_loc = row.get(
                    "beginning_milepost"
                )

            # ----------------------------------
            # Ending location
            # ----------------------------------

            end_loc = None

            if "ending_cross_street" in row.index:

                end_loc = row.get(
                    "ending_cross_street"
                )

            elif "ending_milepost" in row.index:

                end_loc = row.get(
                    "ending_milepost"
                )

            # ----------------------------------
            # Record
            # ----------------------------------

            all_projects.append({

                "state": state,

                "project_id": project_id,

                "project_name": project_name,

                "road": road,

                "direction": direction,

                "project_type": project_type,

                "description": description,

                "start_date": start,

                "end_date": end,

                "duration_days": duration_days,

                "vehicle_impact": vehicle_impact,

                "begin_location": begin_loc,

                "end_location": end_loc

            })

    # --------------------------------------
    # Build dataframe
    # --------------------------------------

    projects = pd.DataFrame(all_projects)

    # --------------------------------------
    # Combine recurring occurrences
    # --------------------------------------

    projects = (
        projects
        .groupby("project_id")
        .agg({
            "state": "first",
            "project_name": "first",
            "road": "first",
            "direction": "first",
            "project_type": "first",
            "description": "first",
            "start_date": "min",
            "end_date": "max",
            "vehicle_impact": "first",
            "begin_location": "first",
            "end_location": "first"
        })
        .reset_index()
    )

    projects['start_date'] = pd.to_datetime(projects['start_date'], errors = 'coerce', utc = True)
    projects['end_date'] = pd.to_datetime(projects['end_date'], errors = 'coerce', utc = True)

    projects["duration_days"] = (
        projects["end_date"] -
        projects["start_date"]
    ).dt.total_seconds() / 86400

    # --------------------------------------
    # Optional cleanup
    # --------------------------------------

    if remove_short_term_events:

        bad_words = [
            "flagging",
            "detour",
            "shoulder closed",
            "lane closed",
            "ramp closed",
            "entrance ramp",
            "exit ramp"
        ]

        pattern = "|".join(bad_words)

        projects = projects[
            ~projects["description"]
            .fillna("")
            .str.lower()
            .str.contains(pattern)
        ]

    if min_duration_days is not None:

        projects = projects[
            projects["duration_days"] >= min_duration_days
        ]

    # --------------------------------------
    # Sort longest projects first
    # --------------------------------------

    projects = projects.sort_values(
        "duration_days",
        ascending=False
    )

    projects.to_csv(
        output_file,
        index=False
    )

    print()
    print(f"Saved {len(projects):,} projects")
    print(f"Output: {output_file}")

    return projects

build_project_database(
    {
        'MI': 'MI_construction.csv',
        'ND': 'ND_construction.csv',
        'WI': 'WI_construction.csv',
        'IA': 'IA_construction.csv'
    },
    output_file='Midwest_Construction_Projects.csv'
)