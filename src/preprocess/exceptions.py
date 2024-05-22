
class NotEnoughProbesInSetError(Exception):
    def __init__(self, set_name: str, probes_diff: int) -> None:
        super().__init__(f"Not enough probes for {set_name}. Missing probes: {probes_diff}")
