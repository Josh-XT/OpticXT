# Contributing to OpticXT

We welcome contributions to OpticXT! This guide will help you get started.

## Development Setup

### Prerequisites

- Rust 1.70+ 
- NVIDIA GPU with CUDA support (recommended)
- Camera and microphone for testing

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Josh-XT/OpticXT.git
cd OpticXT
```

2. Build with CUDA support:
```bash
cargo build --release --features cuda
```

3. Run tests:
```bash
cargo test
```

## Code Style

- Use `cargo fmt` to format code
- Use `cargo clippy` to check for common mistakes
- Follow Rust naming conventions
- Add documentation for public APIs

## Testing

- Add tests for new functionality
- Ensure existing tests pass
- Test both CUDA and CPU modes
- Test with real hardware when possible

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push to your fork
7. Create a pull request

## Areas for Contribution

- **Vision Processing**: Improve object detection and scene understanding
- **Performance**: Optimize GPU utilization and inference speed
- **Hardware Support**: Add support for new cameras and sensors
- **Documentation**: Improve setup guides and API documentation
- **Testing**: Add comprehensive test coverage

## Questions?

Feel free to open an issue for questions or discussions about contributing.
