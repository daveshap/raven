# RAVEN Open Source Software (OSS) Community Project

RAVEN (Realtime Assistant Voice Enabled Network) OSS community edition.

# Overview & Intro

RAVEN is meant to be a digital assistant and artificial cognitive entity. The goal is to create an autonomous artificial intelligence that is reliable, trustworthy, and useful. Here are some of the top priorities for RAVEN:

1. Voice-enabled assistant
2. Extensible capabilities (plug and play cognitive architecture)
3. Fully autonomous (can think and make decisions independently)
4. Long-term memory and continuous learning

[![YouTube Thumbnail and Link to RAVEN Intro](http://img.youtube.com/vi/EwJ1534Gy6g/0.jpg)](http://www.youtube.com/watch?v=EwJ1534Gy6g "What is RAVEN? Overview, Introduction, and Community Update - Friday, February 3, 2023")

# Latest News

1. Please see the updated announcement here: https://github.com/daveshap/raven/discussions/52
   - This covers forming a legal entity, establishing governance, and funding

## Primary Pages

- [Project Wiki:](https://github.com/daveshap/raven/wiki)
- [Discussions:]( https://github.com/daveshap/raven/discussions) 
- [Issues:](https://github.com/daveshap/raven/issues)


Please read the draft [Code of Conduct](https://github.com/daveshap/raven/wiki/Code-of-Conduct-(draft)).

There is also a [Contributors Guide draft](https://github.com/daveshap/raven/blob/1.0.0-dev/docs/Contributing.md) that will eventually be merged with the Code of Conduct.

# Development

We are still in the Forming phase. Please be patient - the RAVEN project was launched February 2, 2023 (so it's brand new)

![Development Phases](https://user-images.githubusercontent.com/878550/217951051-62a1784f-96a2-40c8-a2a3-cf48af7bba8f.png)


## Get Started with Development

The development branching strategy is located [here in the Wiki](https://github.com/daveshap/raven/wiki/Development---Branching-Strategy-and-Process-(draft)).


- Clone this repository using your favorite tool(s).

- Checkout the desired branch to work on. In this example 1.0.0-dev

```
git checkout 1.0.0-dev

```

- Create a Python virtual environment with tools of your choice using the environment.yml file in the root of the project.

- Activate the new environment according to instructions in the terminal window.

- The *microservices* are in the services/microservices directory.

### Documentation Development

Documentation is written in markdown files in the docs directory. 

You can learn more about the Markdown syntax [here](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/about-writing-and-formatting-on-github)

MkDocs is installed in the virtual environment, so you can run a live server for previewing the docs in your browser:

```
mkdocs serve

```
