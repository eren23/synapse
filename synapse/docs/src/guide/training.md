# Training

Synapse includes a training stack built on the `synapse-train`, `synapse-autograd`, and `synapse-optim` crates.

## Trainer API

The `synapse-train` crate provides a high-level training loop:

```rust
use synapse_train::{Trainer, TrainerConfig};
use synapse_nn::Linear;
use synapse_optim::Adam;

let model = Linear::new(784, 10);
let optimizer = Adam::new(model.parameters(), 1e-3);

let config = TrainerConfig {
    epochs: 10,
    batch_size: 32,
    ..Default::default()
};

let trainer = Trainer::new(model, optimizer, config);
trainer.fit(&train_data, &val_data)?;
```

## Autograd

The `synapse-autograd` crate implements tape-based automatic differentiation:

```rust
use synapse_autograd::{Variable, backward};

let x = Variable::new(tensor);
let y = model.forward(&x);
let loss = cross_entropy(&y, &targets);

backward(&loss); // Populates .grad for all parameters
```

Gradients are accumulated on a computation tape and propagated in reverse order.

## Optimizers

Available in `synapse-optim`:

| Optimizer | Description |
|-----------|-------------|
| `SGD` | Stochastic gradient descent with optional momentum |
| `Adam` | Adaptive moment estimation |
| `RMSProp` | Root mean square propagation |

All optimizers support learning rate schedulers (step, cosine, warmup).

## Callbacks

The trainer supports callbacks for controlling the training loop:

```rust
use synapse_train::callbacks::{EarlyStopping, ModelCheckpoint};

let callbacks = vec![
    EarlyStopping::new("val_loss", 5),           // Stop after 5 epochs without improvement
    ModelCheckpoint::new("best.pt", "val_loss"),  // Save best model
];

trainer.fit_with_callbacks(&train_data, &val_data, &callbacks)?;
```

## Examples

The `examples/` directory includes training demos:

- **XOR** -- minimal autograd example with a 2-layer MLP
- **MNIST** -- digit classification with convolutional layers
- **Text classification** -- sentiment analysis with embeddings
- **ViT** -- vision transformer training on image data

Run an example:

```bash
cargo run --example mnist_train --release
```
