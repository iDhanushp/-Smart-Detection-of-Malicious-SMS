import com.android.build.gradle.LibraryExtension

allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory = rootProject.layout.buildDirectory.dir("../../build").get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)

    // Temporary workaround for legacy Android library modules (e.g. telephony 0.2.0)
    // that do not declare an `android.namespace` which is mandatory with AGP 8+.
    // We assign a generated namespace if it is missing so the build can proceed.
    plugins.withId("com.android.library") {
        extensions.configure<com.android.build.gradle.LibraryExtension> {
            if (namespace == null) {
                namespace = "fix.${project.name}"
            }
        }

        tasks.withType<com.android.build.gradle.tasks.ProcessLibraryManifest>().configureEach {
            doFirst {
                val manifestFile = file("src/main/AndroidManifest.xml")
                if (manifestFile.exists()) {
                    val original = manifestFile.readText()
                    val modified = original.replace(Regex("package=\"[^\"]*\""), "")
                    if (original != modified) {
                        manifestFile.writeText(modified)
                        println("Removed obsolete package attribute from "+manifestFile)
                    }
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
