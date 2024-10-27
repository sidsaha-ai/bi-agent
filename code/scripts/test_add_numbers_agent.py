"""
This script tests the adding numbers agent.
"""
import argparse

from agents.add_numbers import AddNumbersAgent


def main(inputs):
    """
    The main function to test running the agent.
    """
    print(inputs)
    agent = AddNumbersAgent()

    result = agent.run(inputs)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompt', type=str, required=True,
    )

    args = parser.parse_args()

    main(args.prompt)
