# eva

## Simulation

### Requirements

- Python 3 (3.9 or newer)
- R (4.0.5 or newer)
- Some Python libs (`pip install -r requirements.txt`)

### Usage

Execute the following Python code.

```python
from simulation import Simulation, LinearLine, Point, GeneralLine, ModelLines

Simulation(
    # Required: Force-velocity curve function(s)
    lines=[
        # Line#0
        LinearLine.create(
            # Ratio that this line will be used
            ratio=1,
            # Stall force (pN) and maximum velocity (nm/s)
            x_max=8, y_max=6000,
            # Critical points
            points=[
                Point(x=2, y=3000)
            ],
            # Opposing motor force (default: 0 pN)
            additional_load=1.5,
            # Probability of generating opposing motors (0 <= p <= 1) (default: 0)
            additional_load_probability=0.5
        ),
        # Line#1
        GeneralLine(
            ratio=2,
            x_max=8, y_max=20000,
            # Instead of setting the F-v curve with critical points, it is given by a function
            function=ModelLines.normalized_one_state_model_example,
            # Indicate whether this function is normalized
            is_normalized_function=True
        )
    ],
    # Required: Range of velocity values obtained from experiments (nm/s)
    velocity_range=[500, 5000]
).test()
```
