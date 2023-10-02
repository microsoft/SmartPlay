# SmartPlay

SmartPlay is a benchmark for Large Language Models (LLMs). It is designed to be easy to use, and to provide a wide variety of games to test agents on.

Currently included games are:
- Rock-Paper-Scissors
- 2-armed bandit
- Tower of Hanoi
- [Messenger-EMMA](https://github.com/ahjwang/messenger-emma)
- [Crafter](https://github.com/danijar/crafter)
- [MineDojo Creative Tasks](https://github.com/MineDojo/MineDojo/tree/main)

## Getting Started

SmartPlay requires MineDojo, please follow the official [documentation](https://docs.minedojo.org/sections/getting_started/install.html#direct-install) to install MineDojo first before proceeding.

Then run

```
pip install -e .
```

For completeness, we also provide conda environment scripts and requirements.txt in the root directory.

## Using SmartPlay

Guidelines to use the benchmark are provided in:

```
examples/experiments.py
```

To see all games available in the SmartPlay benchmark, run the following code:

```
import smartplay
print(smartplay.env_list)
```

See [MineDojo Documentation](https://github.com/MineDojo/MineDojo/blob/main/minedojo/tasks/description_files/creative_tasks.yaml) for a description of the MineDojo Creative tasks.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.

