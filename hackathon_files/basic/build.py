"""Regenerate and verify the 50 controller and 10 network-action snapshots."""
from hashlib import sha256
import json
from pathlib import Path

from VeraGridEngine.api import save_file
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from hackathon_files.basic.cases import NetworkSpec, build_network, make_variant, network_specs
from hackathon_files.basic.validation import validate_case


def main(resume: bool = False) -> int:
    """Save five variants of each network and record measured acceptance.

    :param resume: Reuse already verified checkpoint files with matching hashes.
    :return: Zero only after every serialized case passes verification.
    """
    directory: Path = Path(__file__).with_name('testgrids')
    directory.mkdir(parents=True, exist_ok=True)
    report: Path = Path(__file__).with_name('validation.json')
    records: list[dict] = json.loads(report.read_text()) if resume and report.exists() else list()
    position: int = 0
    spec: NetworkSpec
    for spec in network_specs():
        base: MultiCircuit = build_network(spec)
        variant: int
        for variant in range(5):
            name: str = f'basic_{spec.name}_{variant + 1:02d}'
            path: Path = directory / f'{name}.veragrid'
            source_hash: str = sha256((Path(__file__).resolve().parents[2] / spec.source).read_bytes()).hexdigest()
            if position < len(records):
                # Resume trusts only the exact saved and source bytes already
                # measured; a changed or missing checkpoint must be rebuilt.
                previous: dict = records[position]
                if (previous['file'] != path.name or not path.exists()
                        or previous['sha256'] != sha256(path.read_bytes()).hexdigest()
                        or previous['source_sha256'] != source_hash
                        or 'action_catalog' not in previous):
                    raise ValueError(f'{name}: checkpoint differs; rebuild without resume')
                else:
                    print(f'{position + 1:02d}/50 {name}: reused verified checkpoint', flush=True)
            else:
                grid: MultiCircuit = make_variant(base, spec.name, variant)
                save_file(grid, str(path))
                record: dict = validate_case(path)
                record['network'] = spec.name
                record['source'] = spec.source
                record['source_sha256'] = source_hash
                record['variant'] = variant + 1
                record['sha256'] = sha256(path.read_bytes()).hexdigest()
                records.append(record)
                report.write_text(json.dumps(records, indent=2) + '\n')
                print(f"{len(records):02d}/50 {grid.name}: unstable Re={record['baseline_max_real']:+.6f}; "
                      f"tuned Re={record['witness_max_real']:+.6f}; load-step PASS", flush=True)
            position += 1
    from hackathon_files.basic.switching import main as build_switching
    build_switching()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
