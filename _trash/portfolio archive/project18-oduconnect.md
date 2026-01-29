---
title: "ODUConnect - Software Engineering Group Project"
excerpt: "ODUConnect is a web application built on a MSSQL Database / Flask API / ReactJS Front End stack. I let the front end and API development and built the continuous deployment, among other responsibilities.<br/><img src='/images/oduconnect/ODU_connect.png'>"
collection: portfolio
---

ODUConnect is a web application built on a MSSQL Database / Flask API / ReactJS Front End stack. I led the front-end development, developed the API, implemented continuous deployment, and participated in the application design.

The links on this page will take you to parts of our team website, prototype application, and repositories.

## Project Management

I led the front-end development for this project. This involved setting up the development environment, building the CD pipeline, ensuring that each team member was able to learn the framework and participate, as well as breaking the work down into tasks to assign to the team.

## Team Website

The first task in the project was to deploy our [team website](https://www.cs.odu.edu/~411yello/){:target="_blank"}. Because we had a large group for the project and I wanted website edits to be accessible to everyone, I chose the Jekyll framework. Jekyll compiles Markdown into a static HTML website.

![png](/images/oduconnect/team_website.png)

I build a [continuous deployment (GitLab link)](https://git-community.cs.odu.edu/411yellow/website/blob/master/.gitlab-ci.yml){:target="_blank"} pipeline in our repository that automatically deploys the compiled website to our server. The end result was that team members were able to edit the Markdown source files directly via the GitLab repository web interfact and have those changes reflected on the live site a minute later. This achieved 100% participation on team website editing.

## API

I built the [REST API](https://git-community.cs.odu.edu/411yellow/api/blob/master/app.py){:target="_blank"} in Flask because I felt it was a powerful tool that would enable rapid prototype development. I implemented it as a service on the server hosting our MSSQL database. The API connects to the database via SQLAlchemy and calls stored prodecures. It also performs some error handling and data processing.

The API works two ways, providing the front-end application with information and updating the database based on front-end user input.

## Front-End Development

I chose ReactJS for this application because I had a little familiarity with React Native and I felt like this would be the most accessible framework, enabling team members to learn quickly and ensure buy-in for front-end development participation. If the tools are too complex it can be hard to engage the team with development activities, and if it's too simple it won't be powerful enough to accomplish what we need done. (Please ignore the bugs!)

I kicked off the project by deploying the application framework and creating demo pages highlighting the basic capabilities of ReactJS so that team members could see how it works. Coming from a C++ and Python background, JavaScript development was fairly abstract to me and I had to invest a significant amount of time learning how things work.

I co-developed the [home page](https://www.cs.odu.edu/~411yello/oduconnect/#/){:target="_blank"} and developed the [projects](https://www.cs.odu.edu/~411yello/oduconnect/#/projects){:target="_blank"} and [calendar](https://www.cs.odu.edu/~411yello/oduconnect/#/calendar){:target="_blank"} pages myself. I also created the navbar and a users page that is hidden behind a login.

![png](/images/oduconnect/projects.png)

For demonstration purposes, you have most administration rights as a guest. If you click the projects page, you can click cells to update the values and see changes reflected in the database. You can also browse the [repository](https://git-community.cs.odu.edu/411yellow/front-end){:target="_blank"}

## Takeaways

I really enjoyed this project. Sometimes group projects can be stressful, but I was happy how everything came together and everyone was able to do their part.

This was also my first foray into front-end development. I generally prefer working with data processing, machine learning, and APIs and found ReactJS more challenging than I expected. On a regular project I can write unit tests where I know the inputs and outputs and have a clear objective for how to pass those tests. But with front-end development there are so many things at play and it can be a real challenge tracking down bugs or figuring out why something is or isn't happening.

But overall, this was a fun project and a great captsone course for the CS program.
