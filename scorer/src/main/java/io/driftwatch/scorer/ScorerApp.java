package io.driftwatch.scorer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;

/**
 * Scores feature events with a fixed linear model and publishes prediction events
 * to the topic the drift consumer reads.
 */
public final class ScorerApp {

    private static final Logger LOG = LoggerFactory.getLogger(ScorerApp.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String MODEL_NAME = "churn-risk";
    private static final String MODEL_VERSION = "v7";
    private static final String[] FEATURE_ORDER = {"tenure_days", "sessions_7d", "support_tickets", "plan_tier"};
    private static final double[] WEIGHTS = {-0.0042, -0.031, 0.118, 0.27};
    private static final double BIAS = 0.41;

    private ScorerApp() {
    }

    public static void main(String[] args) {
        String bootstrap = env("KAFKA_BOOTSTRAP", "localhost:9092");
        String inputTopic = env("INPUT_TOPIC", "features.raw");
        String outputTopic = env("OUTPUT_TOPIC", "predictions");
        String groupId = env("GROUP_ID", "driftwatch-scorer");

        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrap);
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, groupId);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");

        Properties producerProps = new Properties();
        producerProps.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrap);
        producerProps.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        producerProps.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        producerProps.put(ProducerConfig.ACKS_CONFIG, "all");
        producerProps.put(ProducerConfig.LINGER_MS_CONFIG, "20");

        LOG.info("scorer starting: {} -> {} on {}", inputTopic, outputTopic, bootstrap);

        try (KafkaConsumer<String, String> consumer = new KafkaConsumer<>(consumerProps);
             KafkaProducer<String, String> producer = new KafkaProducer<>(producerProps)) {

            consumer.subscribe(Collections.singletonList(inputTopic));
            Runtime.getRuntime().addShutdownHook(new Thread(consumer::wakeup));

            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));
                if (records.isEmpty()) {
                    continue;
                }
                for (ConsumerRecord<String, String> record : records) {
                    String payload = scoreRecord(record.value());
                    if (payload == null) {
                        continue;
                    }
                    producer.send(new ProducerRecord<>(outputTopic, record.key(), payload));
                }
                producer.flush();
                consumer.commitSync();
            }
        } catch (org.apache.kafka.common.errors.WakeupException wakeup) {
            LOG.info("scorer stopped");
        }
    }

    /**
     * Turns one raw feature event into a prediction event, or null when the event is unusable.
     */
    static String scoreRecord(String raw) {
        try {
            JsonNode node = MAPPER.readTree(raw);
            JsonNode featuresNode = node.path("features");
            if (!featuresNode.isObject()) {
                LOG.warn("event without features object, skipping");
                return null;
            }
            Map<String, Double> features = new java.util.HashMap<>();
            Iterator<Map.Entry<String, JsonNode>> it = featuresNode.fields();
            while (it.hasNext()) {
                Map.Entry<String, JsonNode> entry = it.next();
                if (entry.getValue().isNumber()) {
                    features.put(entry.getKey(), entry.getValue().asDouble());
                }
            }
            for (String name : FEATURE_ORDER) {
                if (!features.containsKey(name)) {
                    LOG.warn("missing feature {}, skipping event", name);
                    return null;
                }
            }

            long started = System.nanoTime();
            double probability = score(features, WEIGHTS, BIAS);
            long latencyMicros = (System.nanoTime() - started) / 1_000L;

            ObjectNode out = MAPPER.createObjectNode();
            out.put("event_id", node.path("event_id").asText(""));
            out.put("model", MODEL_NAME);
            out.put("model_version", MODEL_VERSION);
            out.put("prediction", probability);
            out.put("latency_us", latencyMicros);
            out.put("scored_at", Instant.now().toString());
            out.set("features", featuresNode);
            return MAPPER.writeValueAsString(out);
        } catch (Exception e) {
            LOG.warn("could not score event: {}", e.getMessage());
            return null;
        }
    }

    /**
     * Logistic score of the ordered feature vector against the model weights.
     */
    static double score(Map<String, Double> features, double[] weights, double bias) {
        if (weights.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException("weights must match the feature order");
        }
        double z = bias;
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            Double value = features.get(FEATURE_ORDER[i]);
            if (value == null) {
                throw new IllegalArgumentException("missing feature " + FEATURE_ORDER[i]);
            }
            z += weights[i] * value;
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static String env(String name, String fallback) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? fallback : value;
    }
}
