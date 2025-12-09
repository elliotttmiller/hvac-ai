# Contributing to HVAC AI Platform

Thank you for your interest in contributing to the HVAC AI Platform! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/hvac-ai.git
   cd hvac-ai
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure environment variables**
   - Copy `.env.example` to `.env.local`
   - Update with your configuration

4. **Start development servers**
   ```bash
   ./scripts/dev.sh
   # Or separately:
   npm run dev                      # Frontend
   python hvac_analysis_service.py  # Backend
   ```

## Project Structure

```
hvac-ai/
├── src/                    # Frontend (Next.js/React/TypeScript)
│   ├── app/               # Next.js App Router
│   ├── components/        # React components
│   └── lib/               # Utilities and services
├── python-services/        # Backend (FastAPI/Python)
│   └── core/              # Business logic modules
├── docs/                   # Documentation
├── scripts/               # Utility scripts
└── notebooks/             # Jupyter notebooks
```

## Coding Standards

### TypeScript/JavaScript
- Use TypeScript for all new code
- Follow the existing code style (enforced by Biome)
- Use functional components and hooks in React
- Format code: `npm run format`
- Lint code: `npm run lint`

### Python
- Follow PEP 8 style guide
- Use type hints for function signatures
- Use 4 spaces for indentation
- Maximum line length: 120 characters
- Use docstrings for modules, classes, and functions

### General Guidelines
- Write clear, descriptive commit messages
- Keep commits focused and atomic
- Add comments for complex logic
- Update documentation when adding features

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, maintainable code
   - Add tests if applicable
   - Update documentation

3. **Test your changes**
   ```bash
   npm run lint              # Frontend linting
   npm run build            # Frontend build test
   # Test backend manually
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   **Commit message format:**
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes (formatting)
   - `refactor:` Code refactoring
   - `test:` Adding tests
   - `chore:` Maintenance tasks

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Keep PRs focused and reasonably sized
- Respond to review feedback promptly

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Areas for Contribution

- **Frontend**: UI/UX improvements, new components
- **Backend**: API endpoints, AI model improvements
- **Documentation**: Guides, examples, API docs
- **Testing**: Unit tests, integration tests
- **Performance**: Optimization, caching
- **DevOps**: Docker, CI/CD, deployment

## Getting Help

- Check the [documentation](docs/README.md)
- Review existing [issues](https://github.com/elliotttmiller/hvac-ai/issues)
- Ask questions in pull requests

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to HVAC AI Platform! 🎉
