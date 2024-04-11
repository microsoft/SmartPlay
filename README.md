# SmartPlay

![SmartPlay Teaser](assets/img/teaser.png)

The SmartPlay repository is a benchmark and methodology for evaluating the abilities of large language models (LLMs) as agents. It consists of six different games, including Rock-Paper-Scissors, Tower of Hanoi, and Minecraft, each featuring a unique setting that provides up to 20 evaluation settings and infinite environment variations. The games in SmartPlay challenge a subset of nine important capabilities of an intelligent LLM agent, including reasoning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. The distinction between the set of capabilities each game tests allows for the analysis of each capability separately. SmartPlay serves as a rigorous testing ground for evaluating the overall performance of LLM agents and as a roadmap for identifying gaps in current methodologies.

Currently included games are:
- Rock-Paper-Scissors
- 2-armed bandit
- Tower of Hanoi
- [Messenger-EMMA](https://github.com/ahjwang/messenger-emma)
- [Crafter](https://github.com/danijar/crafter)
- [MineDojo Creative Tasks](https://github.com/MineDojo/MineDojo/tree/main)

For more information, please refer to the [paper](https://arxiv.org/abs/2310.01557).

## Table of Contents

- [Introduction](#introduction)
- [Games in SmartPlay](#games-in-smartplay)
- [Getting Started](#getting-started)
- [Using SmartPlay](#using-smartplay)
- [Citing SmartPlay](#citing-smartplay)
- [Contributing](#contributing)
- [Legal Notices](#legal-notices)

## Games in SmartPlay <a name="games-in-smartplay"></a>
![Games in SmartPlay](assets/figures/fig2.png)

## Getting Started <a name="getting-started"></a>

First consider setting up a conda environment by running 
```
conda env create --name SmartPlay --file environment.yml
```

SmartPlay requires MineDojo, please follow the official [documentation](https://docs.minedojo.org/sections/getting_started/install.html#direct-install) to install MineDojo first before proceeding.

Then run

```bash
pip install -e .
```

For completeness, we also provide conda environment scripts and requirements.txt in the root directory.

## Using SmartPlay <a name="using-smartplay"></a>

SmartPlay is designed to be used with OpenAI Gym:
```python
import gym
import smartplay
env = gym.make("smartplay:{}-v0".format(env_name))
_, info = env.reset()

while True:
    action = info['action_space'].sample()
    _, reward, done, info = env.step(action)
    manual, obs, history, score = info['manual'], info['obs'], info['history'], info['score']
    if not done:
        completion=0
    else:
        completion=info['completed']
```

Full example to use the benchmark are provided in:

```python
examples/experiments.py
```

To see all environments available in the SmartPlay benchmark, run the following code:

```python
import smartplay
print(smartplay.env_list)
```

See [MineDojo Documentation](https://github.com/MineDojo/MineDojo/blob/main/minedojo/tasks/description_files/creative_tasks.yaml) for a description of the MineDojo Creative tasks.

## Citing SmartPlay <a name="citing-smartplay"></a>
```
@inproceedings{wu2024smartplay,
  title={SmartPlay: A Benchmark for LLMs as Intelligent Agents},
  author={Wu, Yue and Tang, Xuan and Mitchell, Tom and Li, Yuanzhi},
  booktitle={ICLR},
  year={2024}
}
```

## Contributing <a name="contributing"></a>

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices <a name="legal-notices"></a>

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

