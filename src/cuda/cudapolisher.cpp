/*!
 * @file cudapolisher.cpp
 *
 * @brief CUDA Polisher class source file
 */

#include "cudapolisher.hpp"

#include <cuda_profiler_api.h>

#include <future>  // NOLINT
#include <iostream>
#include <chrono>  // NOLINT

#include "claragenomics/utils/cudautils.hpp"
#include "biosoup/progress_bar.hpp"
#include "biosoup/timer.hpp"

namespace racon {

CUDAPolisher::CUDAPolisher(double q, double e, std::uint32_t w,
    bool trim, std::int8_t m, std::int8_t n, std::int8_t g,
    std::shared_ptr<thread_pool::ThreadPool> thread_pool,
    std::uint32_t cudapoa_batches, bool cuda_banded_alignment,
    std::uint32_t cudaaligner_batches)
        : Polisher(q, e, w, trim, m, n, g, thread_pool)
        , cudapoa_batches_(cudapoa_batches)
        , cudaaligner_batches_(cudaaligner_batches)
        , gap_(g)
        , mismatch_(n)
        , match_(m)
        , cuda_banded_alignment_(cuda_banded_alignment)
{
    claragenomics::cudapoa::Init();
    claragenomics::cudaaligner::Init();

    CGA_CU_CHECK_ERR(cudaGetDeviceCount(&num_devices_));

    if (num_devices_ < 1)
    {
        throw std::runtime_error("No GPU devices found.");
    }

    // std::cerr << "Using " << num_devices_ << " GPU(s) to perform polishing" << std::endl;

    // Run dummy call on each device to initialize CUDA context.
    for(int32_t dev_id = 0; dev_id < num_devices_; dev_id++)
    {
        CGA_CU_CHECK_ERR(cudaSetDevice(dev_id));
        CGA_CU_CHECK_ERR(cudaFree(0));
        std::cerr << "[racon::CUDAPolisher::CUDAPolisher] initialized device "
                  << dev_id << std::endl;
    }

    // std::cerr << "[CUDAPolisher] Constructed." << std::endl;
}

CUDAPolisher::~CUDAPolisher()
{
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

std::vector<uint32_t> CUDAPolisher::calculate_batches_per_gpu(uint32_t batches, uint32_t gpus)
{
    // Bin batches into each GPU.
    std::vector<uint32_t> batches_per_gpu(gpus, batches / gpus);

    for(uint32_t i = 0; i < batches % gpus; ++i)
    {
        ++batches_per_gpu[i];
    }

    return batches_per_gpu;
}


void CUDAPolisher::FindBreakPoints(
    const std::vector<std::unique_ptr<Overlap>>& overlaps,
    const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
    const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences)
{
    if (cudaaligner_batches_ < 1)
    {
        // TODO: Kept CPU overlap alignment right now while GPU is a dummy implmentation.
        Polisher::FindBreakPoints(overlaps, targets, sequences);
    }
    else
    {
        // TODO: Experimentally this is giving decent perf
        const uint32_t MAX_ALIGNMENTS = 200;

        biosoup::Timer timer;
        timer.Start();

        std::mutex mutex_overlaps;
        uint32_t next_overlap_index = 0;

        // Lambda expression for filling up next batch of alignments.
        auto fill_next_batch = [&mutex_overlaps, &next_overlap_index, &overlaps, &targets, &sequences, this](CUDABatchAligner* batch) -> std::pair<uint32_t, uint32_t> {
            batch->reset();

            // Use mutex to read the vector containing windows in a threadsafe manner.
            std::lock_guard<std::mutex> guard(mutex_overlaps);

            uint32_t initial_count = next_overlap_index;
            uint32_t count = overlaps.size();
            while(next_overlap_index < count)
            {
                if (batch->addOverlap(overlaps.at(next_overlap_index).get(), targets, sequences))
                {
                    next_overlap_index++;
                }
                else
                {
                    break;
                }
            }
            return {initial_count, next_overlap_index};
        };

        std::mutex mutex_bar;
        biosoup::ProgressBar bar{static_cast<std::uint32_t>(overlaps.size()), 16};

        // Lambda expression for processing a batch of alignments.
        auto process_batch = [&fill_next_batch, &mutex_bar, &bar, &timer, this](CUDABatchAligner* batch) -> void {
            while(true)
            {
                auto range = fill_next_batch(batch);
                if (batch->hasOverlaps())
                {
                    // Launch workload.
                    batch->alignAll();
                    batch->find_breaking_points(w_);

                    // progress bar
                    {
                        std::lock_guard<std::mutex> guard(mutex_bar);
                        auto updates = range.second - range.first;
                        while (updates--) {
                          if (++bar) {
                            std::cerr << "[racon::CUDAPolisher::Initialize] aligned overlaps "
                                      << "[" << bar << "] "
                                      << timer.Lap() << "s"
                                      << "\r";
                          }
                        }
                    }
                }
                else
                {
                    break;
                }
            }
        };

        timer.Start();

        // Bin batches into each GPU.
        std::vector<uint32_t> batches_per_gpu = calculate_batches_per_gpu(cudaaligner_batches_, num_devices_);

        for(int32_t device = 0; device < num_devices_; device++)
        {
            for(uint32_t batch = 0; batch < batches_per_gpu.at(device); batch++)
            {
                batch_aligners_.emplace_back(createCUDABatchAligner(15000, 15000, MAX_ALIGNMENTS, device));
            }
        }

        std::cerr << "[racon::CUDAPolisher::initialize] allocated memory on GPUs for alignment "
                  << timer.Stop() << "s"
                  << std::endl;

        timer.Start();

        // Run batched alignment.
        std::vector<std::future<void>> thread_futures;
        for(auto& aligner : batch_aligners_)
        {
            thread_futures.emplace_back(
                    thread_pool_->Submit(
                        process_batch,
                        aligner.get()
                        )
                    );
        }

        // Wait for threads to finish, and collect their results.
        for (const auto& future : thread_futures) {
            future.wait();
        }
        std::cerr << std::endl;

        batch_aligners_.clear();
    }
}

std::vector<std::unique_ptr<biosoup::Sequence>> CUDAPolisher::Polish(
    bool drop_unpolished_sequences)
{
    if (cudapoa_batches_ < 1)
    {
        return Polisher::Polish(drop_unpolished_sequences);
    }
    else
    {
        biosoup::Timer timer;
        timer.Start();
        // Creation and use of batches.
        const uint32_t MAX_DEPTH_PER_WINDOW = 200;

        // Bin batches into each GPU.
        std::vector<uint32_t> batches_per_gpu = calculate_batches_per_gpu(cudapoa_batches_, num_devices_);

        for(int32_t device = 0; device < num_devices_; device++)
        {
            size_t total = 0, free = 0;
            CGA_CU_CHECK_ERR(cudaSetDevice(device));
            CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
            // Using 90% of available memory as heuristic since not all available memory can be used
            // due to fragmentation.
            size_t mem_per_batch = 0.9 * free/batches_per_gpu.at(device);
            for(uint32_t batch = 0; batch < batches_per_gpu.at(device); batch++)
            {
                batch_processors_.emplace_back(createCUDABatch(MAX_DEPTH_PER_WINDOW, device, mem_per_batch, gap_, mismatch_, match_, cuda_banded_alignment_));
            }
        }

        std::cerr << "[racon::CUDAPolisher::Polish] allocated memory on GPUs for polishing "
                  << timer.Stop() << "s"
                  << std::endl;

        // Mutex for accessing the vector of windows.
        std::mutex mutex_windows;

        // Initialize window consensus statuses.
        window_consensus_status_.resize(windows_.size(), false);

        // Index of next window to be added to a batch.
        uint32_t next_window_index = 0;

        // Lambda function for adding windows to batches.
        auto fill_next_batch = [&mutex_windows, &next_window_index, this](CUDABatchProcessor* batch) -> std::pair<uint32_t, uint32_t> {
            batch->reset();

            // Use mutex to read the vector containing windows in a threadsafe manner.
            std::lock_guard<std::mutex> guard(mutex_windows);

            // TODO: Reducing window wize by 10 for debugging.
            uint32_t initial_count = next_window_index;
            uint32_t count = windows_.size();
            while(next_window_index < count)
            {
                if (batch->addWindow(windows_.at(next_window_index)))
                {
                    next_window_index++;
                }
                else
                {
                    break;
                }
            }

            return {initial_count, next_window_index};
        };

        biosoup::ProgressBar bar{static_cast<std::uint32_t>(windows_.size()), 16};
        std::mutex mutex_bar;

        timer.Start();

        // Lambda function for processing each batch.
        auto process_batch = [&fill_next_batch, &mutex_bar, &bar, &timer, this](CUDABatchProcessor* batch) -> void {
            while(true)
            {
                std::pair<uint32_t, uint32_t> range = fill_next_batch(batch);
                if (batch->hasWindows())
                {
                    // Launch workload.
                    const std::vector<bool>& results = batch->generateConsensus();

                    // Check if the number of batches processed is same as the range of
                    // of windows that were added.
                    if (results.size() != (range.second - range.first))
                    {
                        throw std::runtime_error("Windows processed doesn't match \
                                range of windows passed to batch\n");
                    }

                    // Copy over the results from the batch into the per window
                    // result vector of the CUDAPolisher.
                    for(uint32_t i = 0; i < results.size(); i++)
                    {
                        window_consensus_status_.at(range.first + i) = results.at(i);
                    }

                    // progress bar
                    {
                        std::lock_guard<std::mutex> guard(mutex_bar);
                        auto updates = results.size();
                        while (updates--) {
                          if (++bar) {
                            std::cerr << "[racon::CUDAPolisher::Polish] generating consensus "
                                      << "[" << bar << "] "
                                      << timer.Lap() << "s"
                                      << "\r";
                          }
                        }
                    }
                }
                else
                {
                    break;
                }
            }
        };

        // Process each of the batches in a separate thread.
        std::vector<std::future<void>> thread_futures;
        for(auto& batch_processor : batch_processors_)
        {
            thread_futures.emplace_back(
                    thread_pool_->Submit(
                        process_batch,
                        batch_processor.get()
                        )
                    );
        }

        // Wait for threads to finish, and collect their results.
        for (const auto& future : thread_futures) {
            future.wait();
        }
        std::cerr << std::endl;
        timer.Stop();

        // Start timing CPU time for failed windows on GPU
        timer.Start();
        // Process each failed windows in parallel on CPU
        std::vector<std::future<bool>> thread_failed_windows;
        for (uint64_t i = 0; i < windows_.size(); ++i) {
            if (window_consensus_status_.at(i) == false)
            {
                thread_failed_windows.emplace_back(thread_pool_->Submit(
                  [&] (uint64_t j) -> bool {
                    auto it = thread_pool_->thread_ids().find(std::this_thread::get_id());
                    if (it == thread_pool_->thread_ids().end()) {
                      fprintf(stderr, "[racon::CUDAPolisher::Polish] error: "
                          "thread identifier not present!\n");
                      exit(1);
                    }
                    return window_consensus_status_.at(j) =
                        windows_[j]->GenerateConsensus(alignment_engines_[it->second], trim_);
                  }, i));
            }
        }

        // Wait for threads to finish, and collect their results.
        for (const auto& t : thread_failed_windows) {
            t.wait();
        }
        if (thread_failed_windows.size() > 0)
        {
            std::cerr << "[racon::CUDAPolisher::Polish] polished remaining windows on CPU "
                      << timer.Stop() << "s"
                      << std::endl;
        }

        // Collect results from all windows into final output.
        std::string polished_data = "";
        uint32_t num_polished_windows = 0;

        biosoup::Sequence::num_objects = 0;
        std::vector<std::unique_ptr<biosoup::Sequence>> dst;

        timer.Start();

        for (uint64_t i = 0; i < windows_.size(); ++i) {

            num_polished_windows += window_consensus_status_.at(i) == true ? 1 : 0;
            polished_data += windows_[i]->consensus();

            if (i == windows_.size() - 1 || windows_[i + 1]->rank() == 0) {
                double polished_ratio = num_polished_windows /
                    static_cast<double>(windows_[i]->rank() + 1);

                if (!drop_unpolished_sequences || polished_ratio > 0) {
                    std::string tags = " LN:i:" + std::to_string(polished_data.size());
                    tags += " RC:i:" + std::to_string(headers_[windows_[i]->id()]->id);
                    tags += " XC:f:" + std::to_string(polished_ratio);
                    dst.emplace_back(new biosoup::Sequence(
                        headers_[windows_[i]->id()]->name + tags, polished_data));
                }

                num_polished_windows = 0;
                polished_data.clear();
            }
            windows_[i].reset();
        }

        std::cerr << "[racon::CUDAPolisher::Polish] generated consensus "
                  << timer.Stop() << "s"
                  << std::endl;

        // Clear POA processors.
        batch_processors_.clear();
        return dst;
    }
}

}  // namespace racon
