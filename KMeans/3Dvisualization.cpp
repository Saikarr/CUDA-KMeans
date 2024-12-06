#define GLEW_STATIC
#include <glew.h>
#include <glfw3.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <iostream>

#include "headers.h"

// Function to visualize clusters using OPENGL

// Callback function to adjust the viewport when the window is resized
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

// Vertex and fragment shader source code
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float pointID;

out float id;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    id = pointID;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in float id;
out vec4 FragColor;

vec3 getColorFromID(float id) {
    if (abs(id - 1.0) < 0.01) return vec3(1.0, 0.0, 0.0); // Red
    if (abs(id - 2.0) < 0.01) return vec3(0.0, 1.0, 0.0); // Green
    if (abs(id - 3.0) < 0.01) return vec3(0.0, 0.0, 1.0); // Blue
    if (abs(id - 4.0) < 0.01) return vec3(1.0, 1.0, 0.0); // Yellow
    if (abs(id - 5.0) < 0.01) return vec3(0.0, 1.0, 1.0); // Cyan
    if (abs(id - 6.0) < 0.01) return vec3(1.0, 0.0, 1.0); // Magenta
    if (abs(id - 7.0) < 0.01) return vec3(0.5, 0.5, 0.5); // Gray
    if (abs(id - 8.0) < 0.01) return vec3(0.5, 0.0, 0.0); // Dark Red
    if (abs(id - 9.0) < 0.01) return vec3(0.0, 0.5, 0.0); // Dark Green
    if (abs(id - 10.0) < 0.01) return vec3(0.0, 0.0, 0.5); // Dark Blue
    if (abs(id - 11.0) < 0.01) return vec3(0.8, 0.3, 0.0); // Orange
    if (abs(id - 12.0) < 0.01) return vec3(0.3, 0.8, 0.0); // Lime
    if (abs(id - 13.0) < 0.01) return vec3(0.0, 0.3, 0.8); // Sky Blue
    if (abs(id - 14.0) < 0.01) return vec3(0.8, 0.0, 0.3); // Pink
    if (abs(id - 15.0) < 0.01) return vec3(0.3, 0.0, 0.8); // Purple
    if (abs(id - 16.0) < 0.01) return vec3(0.0, 0.8, 0.3); // Sea Green
    if (abs(id - 17.0) < 0.01) return vec3(0.8, 0.8, 0.8); // Light Gray
    if (abs(id - 18.0) < 0.01) return vec3(0.2, 0.2, 0.2); // Dark Gray
    if (abs(id - 19.0) < 0.01) return vec3(1.0, 0.5, 0.5); // Soft Red
    if (abs(id - 20.0) < 0.01) return vec3(0.5, 1.0, 0.5); // Soft Green
    return vec3(1.0, 1.0, 1.0); // Default White
}

void main() {
    FragColor = vec4(getColorFromID(id), 1.0);
}
)";

float lastX = 400, lastY = 300; // Center of the window
float pitch = 0.0f, yaw = 0.0f;
bool firstMouse = true;
bool isLeftMouseButtonPressed = false;

// Mouse movement callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {

	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;
	if (!isLeftMouseButtonPressed)
		return;
	float sensitivity = 0.1f; // Adjust sensitivity
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// Constrain pitch to avoid gimbal lock
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;
}

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			isLeftMouseButtonPressed = true;
		}
		else if (action == GLFW_RELEASE) {
			isLeftMouseButtonPressed = false;
		}
	}
}

// Function to visualize clusters
void VisualizeClusters(std::vector<float> pointsAndAssignments) {

	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Create a window
	GLFWwindow* window = glfwCreateWindow(800, 600, "Visualize 3D Points", nullptr, nullptr);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Set viewport and callback
	glViewport(0, 0, 800, 600);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);


	// Create and bind the Vertex Array Object and Vertex Buffer Object
	unsigned int VAO, VBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, pointsAndAssignments.size() * sizeof(float), pointsAndAssignments.data(), GL_STATIC_DRAW);

	// Vertex attributes
	// Position (location 0)
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// Point ID (location 1)
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Compile the vertex shader
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);

	// Compile the fragment shader
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);

	// Link shaders into a program
	unsigned int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	// Projection matrix
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

	// Render loop
	while (!glfwWindowShouldClose(window)) {

		glClear(GL_COLOR_BUFFER_BIT);

		// Calculate view matrix based on mouse input
		glm::mat4 view = glm::lookAt(
			glm::vec3(0.0f, 0.0f, 3.0f),                   // Camera position
			glm::vec3(0.0f, 0.0f, 0.0f),                   // Look at the origin
			glm::vec3(0.0f, 1.0f, 0.0f)                    // Up vector
		);
		glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(pitch), glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, glm::radians(yaw), glm::vec3(0.0f, 1.0f, 0.0f));

		// Use the shader program and set uniforms
		glUseProgram(shaderProgram);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

		// Draw points
		glBindVertexArray(VAO);
		glPointSize(5.0f);
		glDrawArrays(GL_POINTS, 0, pointsAndAssignments.size() / 4);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Free resources
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteProgram(shaderProgram);

	glfwTerminate();
}