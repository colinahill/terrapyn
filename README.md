# terrapyn


[![PyPI version](https://badge.fury.io/py/terrapyn.svg)](https://badge.fury.io/py/terrapyn)
![versions](https://img.shields.io/pypi/pyversions/terrapyn.svg)
[![GitHub license](https://img.shields.io/github/license/colinahill/terrapyn.svg)](https://github.com/colinahill/terrapyn/blob/main/LICENSE)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Toolkit to manipulate Earth observations and models.


- Free software: BSD-3-Clause
- Documentation: https://colinahill.github.io/terrapyn.


## Setup

### Turn the project directory into a local Git repository
First, move into the package directory:

```bash
cd project-name/
```

Then, install the poetry dependency manager:

```bash
pip install poetry && poetry install
```

*For more info on poetry, see [python-poetry.org](https://python-poetry.org/).*

Lastly, make the directory a Git repository:

```bash
git init
```

### Link the local repository to a remote repository
Create a remote repository on GitHub for this project with the following steps:
1. Go to [create new repository](https://github.com/new) on GitHub.com.
2. Fill in the repository name (`project-name` from above) and description.
3. Choose a Public or Private repo.
4. Leave all other boxes unchecked.

Copy the URL of the repo and set it as the `origin` remote:

```bash
git remote add origin https://gitlab.com/<username>/<project-name>.git
```

Install pre-commit so we perform checks before committing files

```bash
poetry run pre-commit install
```

Add the files and stage them for commitment

```
git add .
git commit -m "initial commit"
```

Name the current branch to *main*:

```
git branch -M main
```

Finally, push the initial commit to GitHub:

```
git push -u origin main
```

You should now see all the files in the repository on GitHub.

## GitHub Continuous Integration
GitHub Actions takes care of the CI (testing code and deploying docs).

### Setup development branch
Create a development branch where changes will by pushed for testing/development, before it is merged to the `main` branch once it's safe for production.
1. Click on the "main" dropdown button your repo's homepage
2. Type in "development" in the search bar
3. Select "Create branch: development from main".

**Note if 'development' is not used. You must change the branch reference in certain files within the '.github/workflows' directory for CI checks to work.**

As seen with main, Actions will run for `development`. Additionally, whenever you open a pull request (PR) from `development` to `main`, the checks will automatically run to make sure the code is safe for merging. This is the basic requirement for CI to work.

CI checks exist on the `main` and `development` branches so these are typically "protected" from having bad code. Having the test pypi before pypi allows for user-testing of a new release without pushing out code that isn't ready for production. Documentation is auto-published when `main` is updated.

## Create documentation
Github Actions creates a `gh-pages` branch with files that are automatically generated from the repository's `docs` folder. For more information on how to add pages to this documentation visit [MkDocs-Material docs](https://squidfunk.github.io/mkdocs-material/).

To publish these docs to your own GitHub site do the following:
1. Go to "Settings" section.
2. Scroll down to "GitHub Pages" section.
3. Within the "Source" section select `gh-pages` branch and `/(root)`.
4. Click "Save" button.

Scroll back down to "GitHub Pages" and a link should be given for where your docs will be published (wait a few minutes for publication). In addition, this link can be added in the `About` section of your repository under "website" to display the link in a convenient location.

## Publish to Test PyPI

### Release process
The cookiecutter creates a project which has a release action for release candidates and a seperate one for production releases. This is because, before merging into main (and releasing to production) the developer should develop the code on the `development` branch. With the `development` code ready for production, the developer would create a *release candidate* (otherwise known as *pre-release*) which ships the package to [test pypi](https://test.pypi.org). This is done so the package can be downloaded and user-tested without messing up the *release history* of your pypi package. In addition, it's typically recommended to only have production-ready code on your main branch.

For this process to work, we require both pypi and test pypi accounts:

### Create test PyPI account
Visit [test pypi](https://test.pypi.org) and sign-up for an account. Remember your username and password as it will be needed later on.

### Setup repository secrets for test PyPI
Github Secrets allows you to inject secret variables into your Github Actions without them being visible to the public, which is especially important for open-source projects. This cookiecutter's out-of-the-box continous deployment process looks for a few secret variables so we'll set those here.

Navigate to the `Settings -> Secrets` section of your repository and add the secrets `TEST_PYPI_USERNAME` and `TEST_PYPI_PASSWORD`. **If you use other variable names, you'll have to change the secret references in `.github/workflows/test_pypi_publish.yml`.**

### Create release candidate
These are the steps needed to publish to test pypi:

1. Navigate to the `Releases` section of your repository.
2. Click "Create a new release".
3. Set `Tag version` to `0.1.0-rc0`.
4. Select `development` branch.
5. Set `Release title` to `Release Candidate: v0.1.0` (or another name).
6. Select "This is a pre-release"
7. Click "Publish release"

Wait a few minutes (or watch the Github Action) for it to be published then visit your [test pypi projects page](https://test.pypi.org/manage/projects/) to see the release.

### Install package
Now that you code is on test pypi, it can be installed with `pip install` using the extra argument `--extra-index-url`, allowing pip to also check another registry. The full command is:

```bash
pip install your_package_name==0.1.0rc0 --extra-index-url=https://test.pypi.org/simple/`
```

As there is no release candidates (and therefore no `0.1.0rc0`) on pypi. This command would not be able to find the given release without the `extra-index-url`.

## Publish to PyPI

## Differences in the process
Publishing to PyPi is similar to the above process publishing to test pypi. Here, the main differences are outlined:

1. Visit [PyPI](https://www.pypi.org) (not test pypi) and create an account
2. Set the given username and password as `PYPI_USERNAME` and `PYPI_PASSWORD`.
3. For release:
    - Select `main branch`,
    - Set tag version to `0.1.0`,
    - Set release title to `v0.1.0`,
    - **DO NOT** select "This is pre-release".

The `pip` command should be:

```bash
pip install package-name
```

or

```bash
pip install package-name==0.1.0
```

if you wish to pin the version number.
