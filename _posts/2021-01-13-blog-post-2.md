---
title: "A Newbie's Guide to Open Source Software Contributions"
date: 2021-01-13
permalink: /posts/2021/01/blog-post-2/
tags:
  - software engineering
---

We all use open source software. Every time you take out your smartphone and connect to the internet you're executing open source code, at every level from the kernel in your device's operating system to the front end code your browser is rendering. Open source software is a powerful resource and a major driving force behind humanity's rapid development of technology.

And it's something you can participate in and contribute to with a surprisingly low barrier to entry.

I'm going to lay out a quick guide that can get you started in three steps: finding a project with issues that match your skill level and expertise, developing a working knowledge of the project, and making your first pull request.

And if this seems intimidating, it really isn't. Open source developers are some of the most helpful people you'll meet, and many of them will pull out beginner issues for new software engineers to learn how to collaborate with others.

Now let's get started.

# Finding a Project

There are two ways that you can search for a project to contribute to. You can either start by looking for project that interest you and then find issues posted for that project, or you can search open issues for all projects that match your skills. The one thing these strategies have in common is that you're looking for the beginner tag.

(You'll want to open a spreadsheet or text file to keep track of several open issues you find in your search so that you can narrow down to one that is a good fit.)

The first is fairly straightforward: look up projects that you've already used on GitHub, click the issues tab, and add "label: beginner" to the search bar. If you're interested in machine learning, Scikit-Learn is a popular project with active development that you could check out. See the open beginner issues [here](https://github.com/scikit-learn/scikit-learn/issues?q=is%3Aissue+is%3Aopen+label%3A+beginner). Note that most project won't have beginner issues available.

The better strategy for finding issues that match your skillset is to search all open issues. Open up the [advanced search on GitHub](https://github.com/search/advanced) and choose your desired language under "Advanced options" then scroll down to "Issues option" and select the "Open" status.

The key to your search is the issue labels. The first issue you want to search for is "**good first issue**." These are issues posted by people explicitly in a teaching mindset. When I started learning software engineering before I even started on my Computer Science degree, an open source developers who posted an issue with this tag acted as a mentor and walked me through the process, from using Git to creating unit tests. (Shout out to [Lj Miranda](https://github.com/ljvmiranda921) and [wzup](https://github.com/whzup)!)

Here are a list of tags to search for:

* good first issue - Mentioned above.
* beginner - Another popular tag for newbies.
* documentation - Not ready to write code? Documentation is an important part of projects.
* bug - Fixing a bug can be easier than implementing a feature in an unfamiliar project.

Make a list of the issues that look cool. When you've found a decent amount, now you want to dive in and see if this is something you can really tackle. Open up the issue and read it carefully, then locate the code it relates to. How to do this varies greatly. Sometimes there's a link and sometimes it's pretty straightforward to click through the folders and find the module. If you get hung up, just introduce yourself in a comment on the issue and ask for a link so you can check it out.

Once you've found a project that's interesting and you feel confident approaching, it's time to develop a working knowledge of the project.
Develop a Working Knowledge of the Project

This is actually the most difficult step, especially if you don't have experience jumping into an existing project before. You know how it's easy to read a book, yet difficult to write one? Code is the opposite. Writing your own code is easy, reading other people's code is challenging.

The first step? Use the software! Download and run a tutorial, run through an example exercise, or otherwise get simple working experience with it. It's easy to overlook this step, but it's important to see the big picture and really understand what the software is doing and the impact that your contribution will have on it.

Next, dive into the main loop! Or if it doesn't have one and is a library, poke around the modules. You want to see what's happening when this code is run. What the top level classes are, what's imported from various modules and where the code being executed is stored. Tracing these various paths through the code will help you develop a good sense of both the structure of project as well as the programming style of the development team.

Finally, dig into to the module or modules that your issue concerns. Familiarize yourself with the imports, how the classes and functions work, and how the imported code works. You want to get a broad understanding of this so that you can not only make your contribution, but ensure that you create appropriate unit test coverage for the code you are adding and avoid introducing bugs to project.

Also important is to keep an eye out for any inefficiencies, bugs, or areas of improvement. For example, [when I was working on pyswarms](https://jayspeidell.github.io/portfolio/project00-pyswarms/), I noticed that there was a default value for a variable that created unnecessarily large meshes representing objective functions. I also noticed that the color gradient couldn't me customized. I made a separate issue to address fixes to the plotting module.

Now it's time to make you contribution! But keep in mind, coding is only part of it.

# Take On the Issue and Make Your First Pull Request

If you haven't already, introduce yourself to the development team through comments on the issue. Let them know you're joining the project and want to take this issue on. Then open and read the contribution guide.

And read the contribution guide again. Seriously, this is important. It will contain information about how to effectively collaborate with the team and deliver code that matches the rest of the project. In it, the developers will talk about things like code formatters that ensure style consistency, unit test guidelines, the continuous integration pipeline, and more. And if you don't understand anything, look it up!

Now it's time to start programming! This is the fun part. Solve whatever issue you're working on, implement that new feature or squash a bug. Write unit tests if need be. **Just keep in mind that you only work on one issue at a time.** This is good etiquette for collaboration, it not only keeps things organized but makes it easy for the developers leading the project to integrate your contribution. (Feel free to work on issues in separate branches, one at a time means one issue per pull request.)

The end result of your work is going to be a "pull request." This may be an unfamiliar term, because if you've only worked on solo or small group projects in school this is a feature of Git that you don't really need. Pull requests are the core collaborative functionality of Git, and it's how you request that the changes made in your personal fork of the project get merged into a branch of the main repository.

You'll make the pull request, and it will be either accepted and merged into the development branch or it will be rejected with comments on how to improve your code. And your first pull request will probably be rejected, so don't feel bad! This is a learning experience. Read through their comments, fix your code, and try again.

# You Did It!

Congrats! Now it's time to sit back and watch your code merged into the main branch at the next release. People are going to use your code, and you've made a small albeit significant contribution to the development of open source software.

## About the Author

I'm a recent Computer Science graduate. I've built a career in marketing, but I love technology and picked up programming and machine learning as a hobby a few years ago. Decided to go back to school and earned a Computer Science degree with a 3.94 GPA while working full time. Awesome, right? Except for the whole graduating into a pandemic hiring freeze that stopped my career transition in its tracks.

I'm looking for software engineering roles. I prefer back end, stuff like APIs, machine learning, C++ and Python, etc. But the specific frameworks and languages don't matter. Throw me at some challenging problems and I'll get it done!
