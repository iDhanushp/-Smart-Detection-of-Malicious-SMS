
allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// Force all modules (including telephony) to use kotlin-gradle-plugin 1.8.10
buildscript {
    configurations.configureEach {
        resolutionStrategy.eachDependency {
            if (requested.group == "org.jetbrains.kotlin" &&
                requested.name == "kotlin-gradle-plugin") {
                useVersion("1.8.10")
            }
        }
    }
}

val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
    // Force Kotlin version for all subprojects
    plugins.withId("org.jetbrains.kotlin.android") {
        project.extensions.extraProperties["kotlin.version"] = "1.8.10"
    }

    // Also force in buildscript classpath deps
    project.buildscript.configurations.configureEach {
        resolutionStrategy.eachDependency {
            if (requested.group == "org.jetbrains.kotlin" &&
                requested.name == "kotlin-gradle-plugin") {
                useVersion("1.8.10")
            }
        }
    }
}

// Ensure older telephony plugin defines a namespace when building with AGP 8+
subprojects {
    afterEvaluate {
        if (project.name == "telephony") {
            plugins.withId("com.android.library") {
                extensions.configure<com.android.build.gradle.LibraryExtension> {
                    namespace = "com.shounakmulay.telephony"
                }
            }
        }
    }
}

subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
