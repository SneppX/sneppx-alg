"""CLI for listing, viewing, and comparing experiment runs."""

import os
import sys
import json
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="SneppX Experiment Tracker CLI")
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="experiments",
        help="Experiment storage directory",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    list_p = sub.add_parser("list", help="List experiment runs")
    list_p.add_argument(
        "--experiment", type=str, default=None, help="Filter by experiment name"
    )

    # view
    view_p = sub.add_parser("view", help="View a single run")
    view_p.add_argument("run_id", type=str, help="Run ID to view")

    # compare
    cmp_p = sub.add_parser("compare", help="Compare two or more runs")
    cmp_p.add_argument("run_ids", type=str, nargs="+", help="Run IDs to compare")

    # delete
    del_p = sub.add_parser("delete", help="Delete a run")
    del_p.add_argument("run_id", type=str, help="Run ID to delete")

    # export
    exp_p = sub.add_parser("export", help="Export runs to JSON")
    exp_p.add_argument("--output", type=str, default="experiments_export.json")
    exp_p.add_argument("--experiment", type=str, default=None)

    args = parser.parse_args()

    from .experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(storage_dir=args.storage_dir)

    if args.command == "list":
        runs = tracker.list_runs(args.experiment)
        if not runs:
            print("No runs found.")
            return
        print(f"\n{'Run ID':<30} {'Name':<20} {'Status':<12} {'Steps':<8} {'Time':<20}")
        print("-" * 90)
        for r in runs:
            steps = len(next(iter(r.metrics.values()), []))
            t = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(r.start_time))
                if r.start_time
                else ""
            )
            print(f"{r.run_id:<30} {r.run_name:<20} {r.status:<12} {steps:<8} {t:<20}")

    elif args.command == "view":
        run = tracker.get_run(args.run_id)
        if run is None:
            print(f"Run '{args.run_id}' not found.")
            return
        print(f"\nRun ID:       {run.run_id}")
        print(f"Experiment:   {run.experiment_name}")
        print(f"Name:         {run.run_name}")
        print(f"Status:       {run.status}")
        print(
            f"Start:        {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.start_time))}"
            if run.start_time
            else "Start: N/A"
        )
        print(
            f"End:          {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(run.end_time))}"
            if run.end_time
            else "End: N/A"
        )
        print(f"\nParameters:")
        for k, v in run.params.items():
            print(f"  {k}: {v}")
        if run.metrics:
            print(f"\nMetrics:")
            for k, values in run.metrics.items():
                final = values[-1]["value"] if values else None
                print(
                    f"  {k}: {len(values)} logged, last={final:.4f}"
                    if final is not None
                    else f"  {k}: {len(values)} logged"
                )

    elif args.command == "compare":
        runs = []
        for rid in args.run_ids:
            r = tracker.get_run(rid)
            if r is None:
                print(f"Run '{rid}' not found.")
                return
            runs.append(r)
        print(f"\n{'Metric':<25}", end="")
        for r in runs:
            print(f"{r.run_name:<20}", end="")
        print()
        print("-" * (25 + 20 * len(runs)))
        all_metrics = set()
        for r in runs:
            all_metrics.update(r.metrics.keys())
        for metric in sorted(all_metrics):
            print(f"{metric:<25}", end="")
            for r in runs:
                vals = r.metrics.get(metric, [])
                final = vals[-1]["value"] if vals else "-"
                print(f"{str(final):<20}", end="")
            print()

    elif args.command == "delete":
        from .experiment_tracker import LocalBackend

        backend = LocalBackend(args.storage_dir)
        backend.delete_run(args.run_id)
        print(f"Deleted run '{args.run_id}'")

    elif args.command == "export":
        runs = tracker.list_runs(args.experiment)
        data = [r.to_dict() for r in runs]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Exported {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
