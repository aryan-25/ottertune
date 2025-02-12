/*
 * OtterTune - Main.java
 *
 * Copyright (c) 2017-18, Carnegie Mellon University Database Group
 */

package com.controller;

import com.controller.collectors.DBCollector;
import com.controller.collectors.MySQLCollector;
import com.controller.collectors.OracleCollector;
import com.controller.collectors.PostgresCollector;
import com.controller.collectors.SAPHanaCollector;
import com.controller.types.JSONSchemaType;
import com.controller.util.FileUtil;
import com.controller.util.JSONUtil;
import com.controller.util.json.JSONException;
import com.controller.util.json.JSONObject;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.lang.management.ManagementFactory;
import java.util.HashMap;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import sun.misc.Signal;

/**
 * Controller main.
 *
 * @author Shuli
 */
public class Main {
  static final Logger LOG = Logger.getLogger(Main.class);

  // Default output directory name
  private static final String DEFAULT_DIRECTORY = "output";

  // Default observation period time (5 minutes)
  private static final int DEFAULT_TIME_SECONDS = -1;

  // Path to JSON schema directory
  private static final String SCHEMA_PATH = "src/main/java/com/controller/json_validation_schema";

  private static final int TO_MILLISECONDS = 1000;

  private static boolean keepRunning = true;
  private static boolean firstCollecting = false;

  private static int id = 1;

  @SuppressWarnings("restriction")
  public static void main(String[] args) {

    // Initialize log4j
    PropertyConfigurator.configure("log4j.properties");

    // Initialize keepRunning
    keepRunning = true;

    // Initialize firstCollecting
    firstCollecting = false;

    // Create the command line parser
    CommandLineParser parser = new PosixParser();
    Options options = new Options();
    options.addOption("c", "config", true, "[required] Controller configuration file");
    options.addOption("t", "time", true, "The observation time in seconds, default is 300s");
    options.addOption("d","directory", true, "Base directory for the result files, default is 'output'");
    options.addOption("i", "id", true, "A unique identifier of this invocation if multiple instances of the controller are running. Default is 1.");
    options.addOption("h", "help", true, "Print this help");
    String configFilePath = null;

    // Parse the command line arguments
    CommandLine argsLine;
    try {
      argsLine = parser.parse(options, args);
    } catch (ParseException e) {
      LOG.error("Unable to Parse command line arguments");
      printUsage(options);
      return;
    }

    if (argsLine.hasOption("h")) {
      printUsage(options);
      return;
    } else if (argsLine.hasOption("c") == false) {
      LOG.error("Missing configuration file");
      printUsage(options);
      return;
    }

    int time = DEFAULT_TIME_SECONDS;
    if (argsLine.hasOption("t")) {
      time = Integer.parseInt(argsLine.getOptionValue("t"));
    }
    LOG.info("Experiment time is set to: " + time);

    String outputDirectory = DEFAULT_DIRECTORY;
    if (argsLine.hasOption("d")) {
      outputDirectory = argsLine.getOptionValue("d");
    }
    LOG.info("Experiment output directory is set to: " + outputDirectory);
    FileUtil.makeDirIfNotExists(outputDirectory);

    if (argsLine.hasOption("i")) {
      id = Integer.parseInt(argsLine.getOptionValue("i"));
    }
    LOG.info("Experiment id is set to: " + id);

    // Parse controller configuration file
    String configPath = argsLine.getOptionValue("c");
    File configFile = new File(configPath);

    // Check config format
    if (!JSONSchemaType.isValidJson(JSONSchemaType.CONFIG, configFile)) {
      LOG.error("Invalid configuration JSON format");
      return;
    }

    // Load configuration file
    ControllerConfiguration config = null;
    try {
      JSONObject input = new JSONObject(FileUtil.readFile(configFile));
      config =
          new ControllerConfiguration(
              input.getString("database_type"),
              input.getString("username"),
              input.getString("password"),
              input.getString("database_url"),
              input.getString("upload_code"),
              input.getString("upload_url"),
              input.getString("workload_name"));
    } catch (JSONException e) {
      e.printStackTrace();
    }

    DBCollector collector = getCollector(config);

    String pidFileName = id + "_pid.txt";
    File f = new File(pidFileName);
    try {
      // get pid of this process and write the pid to a file before recording the start time
      if (time < 0) {
        // add a signal handler
        Signal.handle(new Signal("INT"), signal -> firstCollecting = true);

        String vmName = ManagementFactory.getRuntimeMXBean().getName();
        int p = vmName.indexOf("@");
        int pid = Integer.valueOf(vmName.substring(0, p));
        try {
          f.createNewFile();
          PrintWriter pidWriter = new PrintWriter(f);
          pidWriter.println(pid);
          pidWriter.flush();
          pidWriter.close();
        } catch (IOException ioe) {
          ioe.printStackTrace();
        }
      }
      else {
        firstCollecting = true;
      }
      LOG.info("Output the process pid to " + pidFileName);

      while (!firstCollecting) {
        Thread.sleep(1);
      }

      // first collection (before queries)
      LOG.info("First collection of metrics before experiment");
      String metricsBefore = collector.collectMetrics();
      if (!JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, metricsBefore)) {
        LOG.error("Invalid output JSON format (metrics_before)");
        return;
      }
      PrintWriter metricsWriter =
          new PrintWriter(FileUtil.joinPath(outputDirectory, "metrics_before.json"), "UTF-8");
      metricsWriter.println(metricsBefore);
      metricsWriter.close();

      String knobs = collector.collectParameters();
      if (!JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, knobs)) {
        LOG.error("Invalid output JSON format (knobs)");
        return;
      }
      PrintWriter knobsWriter =
          new PrintWriter(FileUtil.joinPath(outputDirectory, "knobs.json"), "UTF-8");
      knobsWriter.println(knobs);
      knobsWriter.close();

      // record start time
      long startTime = System.currentTimeMillis();
      LOG.info("Starting the experiment ...");

      // go to sleep
      if (time >= 0) {
        Thread.sleep(time * TO_MILLISECONDS);
      } else {
        // add a signal handler
        Signal.handle(new Signal("INT"), signal -> keepRunning = false);
        while (keepRunning) {
          Thread.sleep(1);
        }
        f.delete();
      }

      long endTime = System.currentTimeMillis();
      long observationTime = time >= 0 ? time : (endTime - startTime) / TO_MILLISECONDS;
      LOG.info("Done running the experiment");

      // summary json obj
      JSONObject summary = null;
      try {
        summary = new JSONObject();
        summary.put("start_time", startTime);
        summary.put("end_time", endTime);
        summary.put("observation_time", observationTime);
        summary.put("database_type", config.getDBName());
        summary.put("database_version", collector.collectVersion());
        summary.put("workload_name", config.getWorkloadName());
      } catch (JSONException e) {
        e.printStackTrace();
      }
      if (!JSONSchemaType.isValidJson(JSONSchemaType.SUMMARY, summary.toString())) {
        LOG.error("Invalid summary JSON format");
        return;
      }

      // write summary JSONObject into a JSON file
      PrintWriter summaryout =
          new PrintWriter(FileUtil.joinPath(outputDirectory, "summary.json"), "UTF-8");
      summaryout.println(JSONUtil.format(summary.toString()));
      summaryout.close();

      // second collection (after workload execution)
      LOG.info("Second collection of metrics after experiment");
      collector = getCollector(config);
      String metricsAfter = collector.collectMetrics();
      if (!JSONSchemaType.isValidJson(JSONSchemaType.OUTPUT, metricsAfter)) {
        LOG.error("Invalid output JSON format (metrics_after)");
        return;
      }
      PrintWriter metricsWriterFinal =
          new PrintWriter(FileUtil.joinPath(outputDirectory, "metrics_after.json"), "UTF-8");
      metricsWriterFinal.println(metricsAfter);
      metricsWriterFinal.close();
    } catch (FileNotFoundException | UnsupportedEncodingException | InterruptedException e) {
      LOG.error("Failed to produce output files");
      e.printStackTrace();
    }
  }

  private static void printUsage(Options options) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp("controller", options);
  }

  private static DBCollector getCollector(ControllerConfiguration config) {
    DBCollector collector = null;
    switch (config.getDBType()) {
      case POSTGRES:
        collector =
            new PostgresCollector(
                config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      case MYSQL:
        collector =
            new MySQLCollector(config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      case SAPHANA:
        collector =
            new SAPHanaCollector(config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      case ORACLE:
        collector =
            new OracleCollector(config.getDBURL(), config.getDBUsername(), config.getDBPassword());
        break;
      default:
        LOG.error("Invalid database type");
        throw new RuntimeException("Invalid database type");
    }
    return collector;
  }
}
